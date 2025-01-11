def calculate_rsi(data, symbol, window=14): 
    delta = data[f'Close_{symbol}'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss

    return 100 - (100 / (1 + rs))
