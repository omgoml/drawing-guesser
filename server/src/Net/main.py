from inference.inference import InferenceModel
from train.TrainingProgress import TrainingProgress
def main():
    IModel = InferenceModel()
    
    try:
        if IModel.model is None and IModel.load_trained_model() == "fail":
            TrainingModel = TrainingProgress() 

            TrainingModel.train()

            print("loading model....")

            IModel.load_trained_model()
        else:
            while True:
                image_path = input("enter your image path (or 'quit' to exis):")

                if image_path.lower() == "quit":
                    break 

                try:
                    prediction, confidence, _ = IModel.prediction_image_path(image_path)
                    print(f"predicted category: {prediction}")
                    print(f"confidence: {confidence:.4f}")
                except Exception as e:
                    print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
