from ultralytics import YOLO
import cv2
import os

def detect_objects(image_path, model_size="yolov8n.pt", save_result=True, show_result=True):
    """
    Detect objects in a single image using YOLOv8
    
    Args:
        image_path: Path to the input image
        model_size: YOLOv8 model size (n, s, m, l, x)
        save_result: Whether to save the annotated image
        show_result: Whether to display the result
    """
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    # Load model
    print(f"Loading model...")
    model = YOLO(model_size)
    
    # Perform detection
    print(f"Detecting objects in {image_path}...")
    results = model(image_path)
    
    # Get detection results
    result = results[0]
    
    # Print detected objects
    print(f"\n{'='*50}")
    print(f"DETECTION RESULTS:")
    print(f"{'='*50}")
    
    if len(result.boxes) == 0:
        print("No objects detected.")
    else:
        for i, box in enumerate(result.boxes, 1):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            
            print(f"{i}. {class_name}: {confidence:.2%} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    
    # Save result
    if save_result:
        output_path = f"bin/detected_{os.path.basename(image_path)}"
        result.save(output_path)
        print(f"\n✓ Saved annotated image to: {output_path}")
    
    # Display result
    if show_result:
        result.show()
        print(f"\n✓ Displaying result (press any key to close)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results

def main():
    """Main function to handle user input"""
    
    print("\n" + "="*50)
    print("YOLOv8 OBJECT DETECTION")
    print("="*50)
    
    # Model options
    print("\nAvailable model sizes:")
    print("1. Nano (fastest, least accurate)")
    print("2. Small (balanced)")
    print("3. Medium (good accuracy)")
    print("4. Large (high accuracy)")
    print("5. Extra Large (maximum accuracy, slower)")
    
    model_choice = input("\nChoose model (1-5, default=1): ").strip()
    model_map = {
        "1": "yolov8n.pt",
        "2": "yolov8s.pt", 
        "3": "yolov8m.pt",
        "4": "yolov8l.pt",
        "5": "yolov8x.pt"
    }
    model_size = model_map.get(model_choice, "yolov8n.pt")
    
    # Get image path from user
    print("\nEnter the path to your image:")
    print("(Examples: image.jpg, C:/Users/Desktop/photo.png, ./images/test.jpg)")
    
    while True:
        image_path = input("Image path: ").strip().strip('"').strip("'")
        
        # Remove quotes if present
        image_path = image_path.strip('"').strip("'")
        
        if os.path.exists(image_path):
            break
        else:
            print(f"File not found: {image_path}")
            print("Please enter a valid image path or type 'quit' to exit.")
            if input().lower() == 'quit':
                return
    
    # Ask for save/display options
    save_choice = input("\nSave annotated image? (y/n, default=y): ").strip().lower()
    save_result = save_choice != 'n'
    
    show_choice = input("Display result? (y/n, default=y): ").strip().lower()
    show_result = show_choice != 'n'
    
    # Run detection
    detect_objects(image_path, model_size, save_result, show_result)

# Alternative: Simple function for quick use
def quick_detect(image_path):
    """
    Quick and simple detection function
    
    Usage: quick_detect("my_image.jpg")
    """
    from ultralytics import YOLO
    
    model = YOLO("yolov8n.pt")
    results = model(image_path)
    
    # Print results
    for box in results[0].boxes:
        class_name = model.names[int(box.cls)]
        confidence = box.conf[0].item()
        print(f"{class_name}: {confidence:.2%}")
    
    results[0].show()  # Display
    results[0].save(f"output_{image_path}")  # Save
    
    return results

if __name__ == "__main__":
    main()
    
    # Example of direct function call (uncomment to use):
    # quick_detect("your_image.jpg")