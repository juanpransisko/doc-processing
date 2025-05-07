from metrics import evaluate_json

def main():
    pred_file = "outputs/test_predictions.json"
    gt_file = "ground_truth/test_labels.json"

    results = evaluate_json(pred_file, gt_file)

    print("\n📊 Evaluation Metrics:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()

