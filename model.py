from fastai.vision.all import *
import pickle
import zipfile

class VoiceFakeDetection:
    def __init__(self):
        self.architectures = {
            'VGG16': vgg16,
            'VGG19': vgg19,
            'ResNet18': resnet18,
            'ResNet34': resnet34,
            'ResNet50': resnet50,
            'alexnet' : alexnet,
        }
        self.transforms = {
            'Resize': Resize((128, 128)),
            'Random Crop': RandomCrop(128),
        }

    def train_model(self, architecture_name, transform_type, zipFile, num_epochs, num_batches, callbacks):
        tmp_path = "/".join(zipFile.split('/')[:-1])
        if zipfile.is_zipfile(zipFile.name):
            with zipfile.ZipFile(zipFile.name, 'r') as zip_ref:
                zip_ref.extractall(tmp_path)
        if architecture_name not in self.architectures:
            return "Architecture not suported."

        if transform_type in self.transforms:
            transform = self.transforms[transform_type]
        else:
            return "Transformation not suported."

        try:
            dls = ImageDataLoaders.from_folder(
                path=tmp_path,
                bs=num_batches,
                valid_pct=0.3,
                item_tfms=transform
            )
        except Exception as e:
            return f"Erro ao carregar dados: {str(e)}"

        try:
            self.model = vision_learner(dls, self.architectures[architecture_name], metrics=F1Score(average='macro'))
            self.model.fine_tune(num_epochs, cbs=eval(callbacks))

            weights_path = f"{architecture_name}_{transform_type}.pkl"
            with open(weights_path, 'wb') as f:
                pickle.dump(self.model, f)

            return f"Training completed! Model save as {weights_path}."
        except Exception as e:
            return f"Error in training: {str(e)}"