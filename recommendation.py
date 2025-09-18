def recommend_models(results):

     # Best accuracy
    best_acc = max(results, key=lambda r: r['r2'])

    # Best speed
    best_speed = min(results, key=lambda r: r['runtime'])

    # Best green (lowest carbon emissions)
    best_green = min(results, key=lambda r: r['co2_kg'])

    # Balanced trade-off
    max_r2 = max(r['r2'] for r in results) or 1
    max_runtime = max(r['runtime'] for r in results) or 1
    max_emissions = max(r['co2_kg'] for r in results) or 1

    for r in results:
        r['score'] = (r['r2'] / max_r2) \
                     - (r['runtime'] / max_runtime) \
                     - (r['co2_kg'] / max_emissions)

    best_balanced = max(results, key=lambda r: r['score'])

    return {
        'best_accuracy': best_acc,
        'best_speed': best_speed,
        'best_green': best_green,
        'best_balanced': best_balanced
    }