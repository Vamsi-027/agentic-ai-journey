PRICING_TABLE = {
    # model pattern -> (input_cost_per_million_tokens, output_cost_per_million_tokens)
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),    # Custom model name used in prompt experiments
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-haiku-4-5": (0.80, 4.00),      # Custom model name used in prompt experiments
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculates the estimated cost in USD for a given model and token metrics."""
    input_cost_per_m, output_cost_per_m = 0.0, 0.0
    
    # Check if model string contains any of our defined model prefixes
    for pattern, prices in PRICING_TABLE.items():
        if pattern in model.lower():
            input_cost_per_m, output_cost_per_m = prices
            break
    else:
        # Fallback values if model is unknown
        if "gpt-4" in model.lower() or "sonnet" in model.lower():
            input_cost_per_m, output_cost_per_m = (3.00, 15.00)
        elif "mini" in model.lower() or "haiku" in model.lower() or "gpt-3.5" in model.lower():
            input_cost_per_m, output_cost_per_m = (0.80, 4.00)

    input_cost = (input_tokens / 1_000_000) * input_cost_per_m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_m
    return round(input_cost + output_cost, 6)
