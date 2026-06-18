from config import Config

def calculate_ema(prices, period):
    """
    Calculates Exponential Moving Average (EMA) for a list of prices.
    Returns a list of same length, with None for indices less than (period - 1).
    """
    if len(prices) < period:
        return [None] * len(prices)
    
    ema = [None] * len(prices)
    
    # Calculate initial SMA for the first 'period' elements
    sma = sum(prices[:period]) / period
    ema[period - 1] = sma
    
    # Calculate multiplier
    multiplier = 2 / (period + 1)
    
    # Calculate EMA for the rest of prices
    for i in range(period, len(prices)):
        ema[i] = (prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier))
        
    return ema

def check_signal(ohlcv_data):
    """
    Evaluates the EMA Crossover strategy on historical candle data.
    ohlcv_data is expected to be a list of lists: [[timestamp, open, high, low, close, volume], ...]
    
    Returns:
        "BUY"  - If short EMA crosses above long EMA on the last closed candle.
        "SELL" - If short EMA crosses below long EMA on the last closed candle.
        "HOLD" - No crossover.
    """
    if not ohlcv_data or len(ohlcv_data) < Config.LONG_EMA + 2:
        return "HOLD"
    
    # Extract closing prices
    closes = [candle[4] for candle in ohlcv_data]
    
    # Calculate short and long EMAs
    ema_short = calculate_ema(closes, Config.SHORT_EMA)
    ema_long = calculate_ema(closes, Config.LONG_EMA)
    
    # Check for valid EMA values
    if ema_short[-3] is None or ema_long[-3] is None or ema_short[-2] is None or ema_long[-2] is None:
        return "HOLD"
    
    # Index -2 is the last completed candle (closed candle)
    # Index -3 is the candle prior to that
    prev_short = ema_short[-3]
    prev_long = ema_long[-3]
    curr_short = ema_short[-2]
    curr_long = ema_long[-2]
    
    # Print calculated EMAs for debugging
    print(f"Strategy Check:")
    print(f"  Closed Price (last): {closes[-2]} | Short EMA ({Config.SHORT_EMA}): {curr_short:.2f} | Long EMA ({Config.LONG_EMA}): {curr_long:.2f}")
    
    # Golden Cross: Short EMA crosses ABOVE Long EMA
    if prev_short <= prev_long and curr_short > curr_long:
        return "BUY"
    
    # Death Cross: Short EMA crosses BELOW Long EMA
    if prev_short >= prev_long and curr_short < curr_long:
        return "SELL"
        
    return "HOLD"
