from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, name="BaseStrategy"):
        self.name = name

    @abstractmethod
    def generate_signal(self, htf_df, ltf_df):
        """
        Analyzes Higher Timeframe (HTF) and Lower Timeframe (LTF) dataframes to output a signal.
        
        Args:
            htf_df: pandas DataFrame containing HTF candles
            ltf_df: pandas DataFrame containing LTF candles
            
        Returns:
            signal: str ("BUY", "SELL", "HOLD")
            metadata: dict containing technical details (e.g., stop_loss, take_profit, indicators)
        """
        pass
