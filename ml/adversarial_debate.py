from config import Config

class AdversarialDebateCourtroom:
    """
    Dual-Brain Adversarial AI Debate Engine (Bull Advocate vs. Bear Prosecutor).
    Stress-tests potential trade setups through adversarial arguments before execution.
    """
    def __init__(self):
        pass

    def conduct_debate(self, signal: str, metadata: dict, market_context: dict) -> dict:
        """
        Runs the adversarial debate between Advocate and Prosecutor.
        Returns verdict ('APPROVED' / 'REJECTED'), net conviction, and argument transcripts.
        """
        if not getattr(Config, 'ENABLE_ADVERSARIAL_DEBATE', True):
            return {
                'verdict': 'APPROVED',
                'conviction_pct': 100.0,
                'advocate_score': 100,
                'prosecutor_score': 0,
                'transcript': 'Adversarial Debate disabled. Auto-approved.'
            }

        is_buy = (signal == "BUY")
        advocate_points = []
        prosecutor_objections = []

        advocate_score = 0
        prosecutor_score = 0

        # ── 1. Advocate Evaluation (Why this trade will WIN) ───────────────────
        # Trend alignment
        if metadata.get('htf_trend') == ('BULLISH' if is_buy else 'BEARISH'):
            advocate_score += 25
            advocate_points.append(f"Strong 1H HTF {metadata.get('htf_trend')} trend alignment (+25pts)")

        # SMC Zone
        setup_type = metadata.get('setup_type', 'NONE')
        if setup_type in ['OB', 'FVG', 'SWEEP']:
            advocate_score += 20
            advocate_points.append(f"Institutional SMC {setup_type} zone confluence (+20pts)")

        # CVD Absorption
        cvd_info = market_context.get('cvd', {})
        absorption = cvd_info.get('absorption', 'NEUTRAL')
        if (is_buy and absorption == 'BULLISH_ABSORPTION') or (not is_buy and absorption == 'BEARISH_ABSORPTION'):
            advocate_score += 20
            advocate_points.append(f"Institutional CVD {absorption} footprint detected (+20pts)")

        # Liquidation Magnet
        liq_info = market_context.get('liquidation', {})
        hunt_signal = liq_info.get('hunt_signal', 'NONE')
        if (is_buy and hunt_signal == 'BULLISH_LIQUIDATION_HUNT') or (not is_buy and hunt_signal == 'BEARISH_LIQUIDATION_HUNT'):
            advocate_score += 20
            advocate_points.append(f"Liquidation Pool Hunt & Reversal confirmed (+20pts)")

        # ML model confirmation
        ml_conf = market_context.get('ml_confidence', 0.5)
        if (is_buy and ml_conf >= 0.60) or (not is_buy and (1.0 - ml_conf) >= 0.60):
            advocate_score += 15
            advocate_points.append(f"Machine Learning bias confirms direction with {ml_conf*100:.1f}% confidence (+15pts)")


        # ── 2. Prosecutor Evaluation (Why this trade is a TRAP) ───────────────
        # RSI Exhaustion
        rsi_val = metadata.get('ltf_rsi', 50.0)
        if is_buy and rsi_val > 68:
            prosecutor_score += 30
            prosecutor_objections.append(f"RSI overbought exhaustion ({rsi_val:.1f} > 68) - Long trap danger (+30pts)")
        elif not is_buy and rsi_val < 32:
            prosecutor_score += 30
            prosecutor_objections.append(f"RSI oversold exhaustion ({rsi_val:.1f} < 32) - Short trap danger (+30pts)")

        # Funding Rate Drag
        funding_rate = market_context.get('funding_rate', 0.0)
        if is_buy and funding_rate > 0.00025: # > +0.025%
            prosecutor_score += 25
            prosecutor_objections.append(f"Elevated positive funding rate ({funding_rate*100:+.4f}%) represents crowded long risk (+25pts)")
        elif not is_buy and funding_rate < -0.00025:
            prosecutor_score += 25
            prosecutor_objections.append(f"Elevated negative funding rate ({funding_rate*100:+.4f}%) represents crowded short squeeze risk (+25pts)")

        # Volatility Squeeze Trap
        if market_context.get('bb_squeeze', False):
            prosecutor_score += 25
            prosecutor_objections.append(f"Active Bollinger Band Squeeze compression risks dual-side fakeout (+25pts)")

        # Spread / Liquidity penalty
        spread_pct = market_context.get('spread_pct', 0.0)
        if spread_pct > 0.0010:
            prosecutor_score += 20
            prosecutor_objections.append(f"Wider spread ({spread_pct*100:.3f}%) degrades risk-reward (+20pts)")


        # ── 3. Courtroom Verdict Arbiter ──────────────────────────────────────
        total_weight = max(1, advocate_score + prosecutor_score)
        net_conviction = max(0.0, (advocate_score - prosecutor_score) / total_weight)
        conviction_pct = round(net_conviction * 100.0, 1)

        min_conviction_req = getattr(Config, 'MIN_AI_CONVICTION_PCT', 0.70) * 100.0
        approved = (conviction_pct >= min_conviction_req) and (advocate_score >= 45) and (prosecutor_score <= 40)
        verdict = "APPROVED" if approved else "REJECTED_BY_AI_COURTROOM"

        transcript = (
            f"⚖️ [AI COURTROOM VERDICT: {verdict}]\n"
            f"  Advocate Conviction : {advocate_score} pts ({', '.join(advocate_points) if advocate_points else 'None'})\n"
            f"  Prosecutor Objections: {prosecutor_score} pts ({', '.join(prosecutor_objections) if prosecutor_objections else 'None'})\n"
            f"  Net Conviction Score: {conviction_pct}% (Threshold: {min_conviction_req}%)"
        )

        return {
            'verdict': verdict,
            'conviction_pct': conviction_pct,
            'advocate_score': advocate_score,
            'prosecutor_score': prosecutor_score,
            'advocate_points': advocate_points,
            'prosecutor_objections': prosecutor_objections,
            'transcript': transcript
        }
