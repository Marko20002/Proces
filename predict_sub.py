import joblib

MODEL_PATH = "math_task_classifier.joblib"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def predict_subject(text: str):
    model = load_model()
    pred = model.predict([text])[0]
    proba = None
    # ако моделот поддржува predict_proba (LogReg го поддржува)
    try:
        proba_vec = model.predict_proba([text])[0]
        labels = model.classes_
        proba = dict(zip(labels, proba_vec))
    except Exception:
        pass
    return pred, proba


if __name__ == "__main__":
    model = load_model()
    print("Model loaded.")

    while True:
        zadaca = input("\nVnesi tekst na zadaca (ili 'exit'): ").strip()
        if zadaca.lower() == "exit":
            break

        label, proba = predict_subject(zadaca)
        print("Predvidena kategorija:", label)
        if proba:
            print("Verojatnosti po klasa:")
            for k, v in proba.items():
                print(f"  {k}: {v:.3f}")
