import argparse
import os
import subprocess

def yolo_pose_train(data, model, pretrained, epochs, imgsz, batch, project, name, device, cos_lr, resume, pose, seed):
    """
    Function to execute the YOLOv8 pose training command.
    """
    command = [
        'yolo', 'pose', 'train',
        f'data={data}',
        f'model={model}',
        f'pretrained={pretrained}',
        f'epochs={epochs}',
        f'imgsz={imgsz}',
        f'batch={batch}',
        f'project={project}',
        f'name={name}',
        f'device={device}',
        f'cos_lr={cos_lr}',
        f'resume={resume}',
        f'pose={pose}',
        f'seed={seed}'
    ]
    print(f"Executing command: {' '.join(command)}")
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 models on a specified dataset.")
    parser.add_argument('--dataset', type=str, default='ap10k', help='Name of the dataset')
    parser.add_argument('--framework', type=str, choices=['yolov8', 'yolov11'], default='yolov8', help='Network framework (yolov8, yolov11)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=-1, help='Batch size (use -1 for auto)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., 0, cpu)')
    parser.add_argument('--models', type=str, help='Comma-separated list of model codes (n, s, m, l, x)')
    parser.add_argument('--no-pretrained', action='store_true', help='Do not use pretrained models')
    parser.add_argument('--pose', type=float, default=40.0, help='Weight ratio for pose loss')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Default settings
    dataset = args.dataset
    framework = args.framework
    epochs = args.epochs
    batch_size = args.batch
    imgsz = args.imgsz
    device = args.device
    cos_lr = True
    resume = True
    pretrained = not args.no_pretrained
    pose = args.pose
    seed = args.seed

    # Set models based on selection
    models = []
    if args.models:
        model_codes = args.models.split(',')
        for model_code in model_codes:
            if model_code in ['n', 's', 'm', 'l', 'x']:
                models.append(f'{framework}{model_code}-pose.yaml')
            else:
                print(f"Warning: Ignoring invalid model code in selection: {model_code}. Valid codes are n, s, m, l, x.")
    else:
        models = [f'{framework}n-pose.yaml', f'{framework}s-pose.yaml', f'{framework}m-pose.yaml', f'{framework}l-pose.yaml', f'{framework}x-pose.yaml']

    if not models:
        print("Error: No valid model selected after processing input. Please choose from n, s, m, l, x, or leave empty to train all.")
        return

    # Loop through each model for the given dataset
    for model_yaml in models:
        pretrained_model = None
        if pretrained:
            pretrained_model = f"./weights/{model_yaml[:-5]}.pt"
            if not os.path.isfile(pretrained_model):
                print(f"Pretrained model {pretrained_model} not found. Skipping...")
                pretrained_model = False

        model_name = f"{dataset}-{model_yaml[:-5]}"
        output_dir = f"./runs/pose/train/{dataset}"

        # Call the training function
        yolo_pose_train(
            data=f"./configs/data/{dataset}.yaml",
            model=f"./configs/models/{dataset}/{model_yaml}",
            pretrained=pretrained_model,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=output_dir,
            name=model_name,
            device=device,
            cos_lr=cos_lr,
            resume=resume,
            pose=pose,
            seed=seed
        )

if __name__ == "__main__":
    main()