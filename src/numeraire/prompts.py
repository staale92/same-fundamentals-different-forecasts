SYSTEM = "You are a financial analyst. Use only the supplied statements. Treat reporting units and explicitly disclosed currency conversions consistently. Do not identify the company or use remembered company outcomes."


TARGET = "whether the next annual filing reports an increase in basic EPS over its own comparable prior-year EPS, using a consistent share basis. That comparable prior-year EPS can differ from the Current figure shown here after restatement or share-basis adjustments"

QUARTERLY_TARGET = (
    "whether basic EPS in the next fiscal quarter increases over its comparable same quarter one year earlier, "
    "on the share basis reported in that outcome filing"
)

# Generated-reasoning request (elicitation check): a short evidence summary, then one JSON object.
REASONING = (
    "Analyze the observed trends and relevant financial ratios. Provide a concise evidence summary (at most 100 words), "
    "then one final JSON object with probability_percent (a number from 0 to 100) and direction (UP if probability_percent "
    "is at least 50, otherwise DOWN) for " + TARGET + "."
)
