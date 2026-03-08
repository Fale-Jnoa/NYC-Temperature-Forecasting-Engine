"""
KNYC Live Nowcaster — Discord Bot
==================================
Posts hourly KNYC temperature predictions to a channel named "Predictions".

Setup
-----
1. Install dependencies:
       pip install "discord.py>=2.0" python-dotenv requests pandas numpy scikit-learn joblib

2. Create a .env file in the same folder as this script:
       DISCORD_TOKEN=your_bot_token_here
       GUILD_ID=your_server_id_here

3. Train models by running KNYC_Nowcaster.ipynb (cells 0–10), which saves:
       knyc_model_daily_high.pkl
       knyc_model_t3h.pkl
       knyc_model_t6h.pkl

4. Run:
       python knyc_discord_bot.py

Behavior
--------
- Posts one nowcast immediately on startup (so you can verify it works).
- Then waits until :54 UTC of the current hour and posts every 60 min after that
  (3 min after KNYC METAR drops at :51).
- Each post includes: current temp, dewpoint, RH, t+3h, t+6h, model high,
  observed high so far today, and the reassessed high (floor-locked once obs
  exceed the model call).
"""

import os
import re
import asyncio
import requests
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone, timedelta

EST = timezone(timedelta(hours=-5))
from io import StringIO

from dotenv import load_dotenv
import discord
from discord.ext import tasks

# ── Credentials ────────────────────────────────────────────────────────────
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GUILD_ID = int(os.getenv("GUILD_ID") or 0)
CHANNEL_NAME  = 'predictions'

if not DISCORD_TOKEN:
    raise RuntimeError('DISCORD_TOKEN not set. Add it to your .env file.')
if not GUILD_ID:
    raise RuntimeError('GUILD_ID not set. Add it to your .env file.')

# ── Model paths ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

print('Loading models...')
model_high = joblib.load(os.path.join(_HERE, 'knyc_model_daily_high.pkl'))
model_t3h  = joblib.load(os.path.join(_HERE, 'knyc_model_t3h.pkl'))
model_t6h  = joblib.load(os.path.join(_HERE, 'knyc_model_t6h.pkl'))
print('✅ Models loaded')

# ── Feature column list (must match KNYC_Nowcaster.ipynb exactly) ──────────
FEATURE_COLS = [
    'tmpf', 'dwpf', 'relh', 'mslp', 'sknt', 'feel_gap',
    'dew_depression', 'td_ratio',
    'mslp_tend_1h', 'mslp_tend_3h', 'mslp_tend_6h',
    'tmp_tend_1h',  'tmp_tend_3h',  'tmp_tend_6h',
    'tmpf_lag_1h',  'tmpf_lag_2h',  'tmpf_lag_3h',
    'tmpf_lag_6h',  'tmpf_lag_12h', 'tmpf_lag_24h',
    'dwpf_lag_1h',  'dwpf_lag_2h',  'dwpf_lag_3h',
    'dwpf_lag_6h',  'dwpf_lag_12h', 'dwpf_lag_24h',
    'tmpf_roll3_mean', 'tmpf_roll6_mean', 'tmpf_roll24_mean',
    'tmpf_roll3_std',  'mslp_roll3_mean',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'hour', 'month', 'dayofyear',
]

# ── IEM website scraper (faster than API) ──────────────────────────────────
def _scrape_latest_ob():
    """Scrape the latest :51 KNYC obs from IEM obhistory page."""
    today_est = datetime.now(EST).strftime('%Y-%m-%d')
    url = (
        'https://mesonet.agron.iastate.edu/sites/obhistory.php'
        f'?date={today_est}&sortdir=desc&windunits=kt'
        '&station=NYC&network=NY_ASOS&metar=0&madis=0'
    )
    try:
        resp = requests.get(url, timeout=15, headers={'Cache-Control': 'no-cache'})
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
    except Exception:
        return None

    # Observation table is the largest table on the page
    obs = max(tables, key=len)
    if isinstance(obs.columns, pd.MultiIndex):
        obs.columns = [' '.join(str(c) for c in col).strip() for col in obs.columns]

    # Sorted desc — first :51 match is the latest observation
    for _, row in obs.iterrows():
        time_str = str(row.iloc[0]).strip()
        if ':51' not in time_str:
            continue
        try:
            # Parse observation time (EST shown on page) → naive UTC
            obs_est = datetime.strptime(f'{today_est} {time_str}', '%Y-%m-%d %I:%M %p')
            obs_utc = obs_est + timedelta(hours=5)

            # Wind column is "NE 3" or "VRB 5" etc — extract numeric speed
            wind_match = re.search(r'(\d+)', str(row.iloc[1]))
            sknt = float(wind_match.group(1)) if wind_match else 0.0

            return {
                'valid': obs_utc,
                'tmpf':  float(row.iloc[5]),
                'dwpf':  float(row.iloc[6]),
                'feel':  float(row.iloc[7]),
                'relh':  float(str(row.iloc[8]).replace('%', '')),
                'mslp':  float(row.iloc[10]),
                'sknt':  sknt,
            }
        except (ValueError, IndexError, TypeError):
            continue
    return None


# ── IEM data fetcher ────────────────────────────────────────────────────────
def fetch_knyc_recent(hours_back: int = 30) -> pd.DataFrame:
    """Fetch the last `hours_back` hours of KNYC :51 ASOS obs from IEM."""
    now   = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_back)
    end   = now + timedelta(hours=1)  # +1h so IEM includes the current hour's :51 obs

    url = (
        'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
        '?station=KNYC&data=tmpf,dwpf,feel,mslp,sknt,relh,metar'
        f'&year1={start.year}&month1={start.month:02d}&day1={start.day:02d}&hour1={start.hour:02d}'
        f'&year2={end.year}&month2={end.month:02d}&day2={end.day:02d}&hour2={end.hour:02d}'
        '&tz=UTC&format=comma&latlon=no&elev=no&missing=M&trace=T&direct=no'
        f'&_nocache={int(now.timestamp())}'
    )
    print(f'[DEBUG] IEM URL: {url}')

    resp = requests.get(url, timeout=20, headers={'Cache-Control': 'no-cache'})
    resp.raise_for_status()

    text = '\n'.join(l for l in resp.text.splitlines() if not l.startswith('#') and l.strip())
    df   = pd.read_csv(StringIO(text))
    df.columns = df.columns.str.strip()
    df['valid'] = pd.to_datetime(df['valid']).dt.tz_localize(None)

    # Keep only scheduled :51 ASOS obs (KNYC's cadence)
    df = df[df['valid'].dt.minute == 51].copy()

    for col in ['tmpf', 'dwpf', 'feel', 'mslp', 'sknt', 'relh']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['mslp'] = df['mslp'].ffill()
    df['sknt'] = df['sknt'].fillna(0.0)
    df['feel'] = df['feel'].fillna(df['tmpf'])

    # Compute relh via Magnus formula where missing
    mask = df['relh'].isna()
    if mask.any():
        a, b = 17.625, 243.04
        T_C = (df.loc[mask, 'tmpf'] - 32) * 5 / 9
        D_C = (df.loc[mask, 'dwpf'] - 32) * 5 / 9
        df.loc[mask, 'relh'] = (
            100 * np.exp(a * D_C / (b + D_C)) / np.exp(a * T_C / (b + T_C))
        )

    # Parse 6-hour max temp from raw METAR remarks (1snTTT group)
    def _parse_6hr_max(metar_text):
        if not isinstance(metar_text, str):
            return np.nan
        m = re.search(r'\b1(0\d{3}|1\d{3})\b', metar_text)
        if m is None:
            return np.nan
        raw = m.group(1)
        sign = -1 if raw[0] == '1' else 1
        temp_c = sign * int(raw[1:]) / 10.0
        return temp_c * 9 / 5 + 32  # convert to °F

    df['mxtmpf_6hr'] = df['metar'].apply(_parse_6hr_max) if 'metar' in df.columns else np.nan

    # Scrape latest obs from IEM website (updates faster than API)
    scraped = _scrape_latest_ob()
    if scraped is not None:
        latest_valid = df['valid'].max() if not df.empty else None
        if latest_valid is None or scraped['valid'] > latest_valid:
            scraped_row = {col: np.nan for col in df.columns}
            scraped_row.update(scraped)
            df = pd.concat([df, pd.DataFrame([scraped_row])], ignore_index=True)
            print(f'[DEBUG] Appended scraped obs: {scraped["valid"]}')

    return df.sort_values('valid').reset_index(drop=True)


# ── Feature engineering ─────────────────────────────────────────────────────
def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature pipeline used during training."""
    df = raw.copy()

    df['feel_gap']       = df['tmpf'] - df['feel']
    df['dew_depression'] = df['tmpf'] - df['dwpf']
    df['td_ratio']       = df['dwpf'] / (df['tmpf'] + 0.001)

    # Convert UTC valid to EST for time features (matches retrained model)
    valid_est = df['valid'] - timedelta(hours=5)
    df['hour']      = valid_est.dt.hour
    df['month']     = valid_est.dt.month
    df['dayofyear'] = valid_est.dt.dayofyear
    df['dayofweek'] = valid_est.dt.dayofweek
    df['hour_sin']  = np.sin(2 * np.pi * df['hour']  / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour']  / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['mslp_tend_1h'] = df['mslp'].diff(1)
    df['mslp_tend_3h'] = df['mslp'].diff(3)
    df['mslp_tend_6h'] = df['mslp'].diff(6)
    df['tmp_tend_1h']  = df['tmpf'].diff(1)
    df['tmp_tend_3h']  = df['tmpf'].diff(3)
    df['tmp_tend_6h']  = df['tmpf'].diff(6)

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'tmpf_lag_{lag}h'] = df['tmpf'].shift(lag)
        df[f'dwpf_lag_{lag}h'] = df['dwpf'].shift(lag)

    df['tmpf_roll3_mean']  = df['tmpf'].shift(1).rolling(3).mean()
    df['tmpf_roll6_mean']  = df['tmpf'].shift(1).rolling(6).mean()
    df['tmpf_roll24_mean'] = df['tmpf'].shift(1).rolling(24).mean()
    df['tmpf_roll3_std']   = df['tmpf'].shift(1).rolling(3).std()
    df['mslp_roll3_mean']  = df['mslp'].shift(1).rolling(3).mean()

    return df


# ── Prediction logic ────────────────────────────────────────────────────────
def get_nowcast() -> dict:
    """Fetch latest obs and return a dict of all prediction values."""
    raw   = fetch_knyc_recent(hours_back=30)
    df_fe = engineer_features(raw)
    ready = df_fe.dropna(subset=FEATURE_COLS)

    if ready.empty:
        raise ValueError('Not enough recent obs to fill all lag features.')

    latest   = ready.iloc[[-1]]
    valid_t  = latest['valid'].iloc[0]
    print(f'[DEBUG] IEM returned {len(raw)} obs, latest :51 METAR used: {valid_t}')
    cur_temp = float(latest['tmpf'].iloc[0])
    cur_dwpf = float(latest['dwpf'].iloc[0])
    cur_relh = float(latest['relh'].iloc[0])

    pred_high = float(model_high.predict(latest[FEATURE_COLS])[0])
    pred_t3h  = float(model_t3h.predict(latest[FEATURE_COLS])[0])
    pred_t6h  = float(model_t6h.predict(latest[FEATURE_COLS])[0])

    # Use EST calendar day for obs high tracking
    today_est = datetime.now(EST).date()
    valid_est = raw['valid'] - timedelta(hours=5)
    today_mask = valid_est.dt.date == today_est

    # Hourly :51 temps for today (EST)
    today_tmpf = raw.loc[today_mask, 'tmpf'].dropna()
    max_tmpf = float(today_tmpf.max()) if not today_tmpf.empty else None

    # 6-hour max temps from METAR remarks for today (EST)
    today_mxt = raw.loc[today_mask, 'mxtmpf_6hr'].dropna() if 'mxtmpf_6hr' in raw.columns else pd.Series(dtype=float)
    max_mxt = float(today_mxt.max()) if not today_mxt.empty else None

    # obs_high = best of hourly temps and 6-hour METAR maxes
    candidates = [v for v in [max_tmpf, max_mxt] if v is not None]
    obs_high = max(candidates) if candidates else None
    reassessed = max(obs_high if obs_high is not None else -999, pred_high)

    return {
        'valid_t':    valid_t,
        'cur_temp':   cur_temp,
        'cur_dwpf':   cur_dwpf,
        'cur_relh':   cur_relh,
        'pred_t3h':   pred_t3h,
        'pred_t6h':   pred_t6h,
        'pred_high':  pred_high,
        'obs_high':   obs_high,
        'reassessed': reassessed,
    }


# ── Discord embed builder ───────────────────────────────────────────────────
def _temp_color(temp_f: float) -> int:
    if temp_f < 32:   return 0x4169E1  # royal blue
    if temp_f < 50:   return 0x00BFFF  # sky blue
    if temp_f < 70:   return 0x00C851  # green
    if temp_f < 85:   return 0xFF8C00  # orange
    return 0xFF2400                     # red


def build_embed(data: dict) -> discord.Embed:
    color   = _temp_color(data['reassessed'])
    obs_ts  = data['valid_t'].replace(tzinfo=timezone.utc).astimezone(EST).strftime('%H:%M EST')
    obs_str = f"{data['obs_high']:.1f}°F" if data['obs_high'] is not None else '—'
    obs_note = '\n*(obs exceeded model)*' if (
        data['obs_high'] is not None and data['obs_high'] > data['pred_high']
    ) else ''

    embed = discord.Embed(
        title       = '🌡️ KNYC Nowcaster Update',
        description = f'Latest METAR: **{obs_ts}** · Central Park, NYC',
        color       = color,
        timestamp   = datetime.now(timezone.utc),
    )
    embed.add_field(name='Current Temp',       value=f"**{data['cur_temp']:.1f}°F**",   inline=True)
    embed.add_field(name='Dewpoint',           value=f"{data['cur_dwpf']:.1f}°F",        inline=True)
    embed.add_field(name='Rel. Humidity',      value=f"{data['cur_relh']:.0f}%",          inline=True)
    embed.add_field(name='t+3h Forecast',      value=f"{data['pred_t3h']:.1f}°F",         inline=True)
    embed.add_field(name='t+6h Forecast',      value=f"{data['pred_t6h']:.1f}°F",         inline=True)
    embed.add_field(name='\u200b',             value='\u200b',                             inline=True)
    embed.add_field(name='Model Daily High',   value=f"{data['pred_high']:.1f}°F",        inline=True)
    embed.add_field(name='Obs High So Far',    value=obs_str,                              inline=True)
    embed.add_field(
        name='Reassessed High',
        value=f"**{data['reassessed']:.1f}°F**{obs_note}",
        inline=True,
    )
    embed.set_footer(text='HistGradientBoosting · KNYC ASOS · IEM')
    return embed


# ── Discord bot ─────────────────────────────────────────────────────────────
intents = discord.Intents.default()
bot     = discord.Client(intents=intents)


async def post_nowcast_to_channel() -> None:
    """Find the Predictions channel and post the current nowcast embed."""
    guild = bot.get_guild((GUILD_ID))
    if guild is None:
        print(f'[ERROR] Guild {(GUILD_ID)} not found — check GUILD_ID in .env')
        return

    channel = discord.utils.get(guild.text_channels, name=CHANNEL_NAME)
    if channel is None:
        print(f'[ERROR] No channel named #{CHANNEL_NAME} found in {guild.name}')
        return

    try:
        data  = get_nowcast()
        embed = build_embed(data)
        await channel.send(embed=embed)
        ts = datetime.now(EST).strftime('%H:%M EST')
        print(f'[{ts}] Posted nowcast  →  reassessed high {data["reassessed"]:.1f}°F')
    except Exception as exc:
        ts = datetime.now(EST).strftime('%H:%M EST')
        print(f'[{ts}] Error: {exc}')
        await channel.send(f'⚠️ Nowcaster error: `{exc}`')


@bot.event
async def on_ready() -> None:
    print(f'✅ Logged in as {bot.user} (ID: {bot.user.id})')
    # Immediate startup post so you know it's working
    await post_nowcast_to_channel()
    if not nowcast_loop.is_running():
        nowcast_loop.start()


@tasks.loop(hours=1)
async def nowcast_loop() -> None:
    await post_nowcast_to_channel()


@nowcast_loop.before_loop
async def before_nowcast_loop() -> None:
    """Align to :59 UTC so we run 8 min after each KNYC :51 METAR."""
    await bot.wait_until_ready()
    now      = datetime.now(timezone.utc)
    next_run = now.replace(minute=59, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(hours=1)
    wait = (next_run - now).total_seconds()
    print(f'Hourly loop aligned — first tick at {next_run.astimezone(EST).strftime("%H:%M EST")} ({wait / 60:.0f} min)')
    await asyncio.sleep(wait)


bot.run(DISCORD_TOKEN)
