from fastai.vision.all import *
import streamlit as st
import os
from context.userContext import getUserContext

class VoiceFakeDetection:
    def __init__(self) -> None:
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

        # sys.stdout = StreamlitLogger()
        getUserContext() # reload user context between sessions(refresh)
        self.save_path = f".tmp/{st.session_state.username}"
        os.makedirs(self.save_path, exist_ok=True)


    def train_model(self, architecture_name, transform_type, dataset, num_epochs, num_batches, callbacks) -> None:
        st.session_state.training_out = st.empty()
        st.session_state.progress =  st.progress(0.0)
        tmp_path="/teamspace/uploads/mini-dataset"
        if architecture_name not in self.architectures:
            st.warning("Architecture not suported.")

        if transform_type in self.transforms:
            transform = self.transforms[transform_type]
        else:
             st.warning("Transformation not suported.")

        try:
            dls = ImageDataLoaders.from_folder(
                path=tmp_path,
                bs=num_batches,
                valid_pct=0.3,
                item_tfms=transform
            )
        except Exception as e:
             st.error(f"Erro ao carregar dados: {str(e)}")

        try:
            self.model_path = f"{self.save_path}/{architecture_name}_{transform_type}"
            os.makedirs(self.model_path, exist_ok=True)
            self.model = vision_learner(dls, self.architectures[architecture_name], metrics=F1Score(average='macro'), path=self.model_path)
            self.model.fine_tune(num_epochs, cbs=[eval(callbacks), LogCallback, CSVLogger])

            self.model.export("model.pkl")
            st.session_state.uploaded_file = f"{self.model_path}/model.pkl"

            st.success(f"Training completed! Model save in {self.model_path}.")

            self.__plot_loss()
        except Exception as e:
            st.error(f"Error in training: {str(e)}")

    def __plot_loss(self):
        st.code(self.model.recorder.metric_names)
        st.code(self.model.recorder.values)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📉 Training & Validation Loss")
            fig, ax = plt.subplots(figsize=(3, 3))
            self.model.recorder.plot_loss(ax=ax)
            st.pyplot(fig, use_container_width=False)

        with col2:
            st.subheader("🔢 Confusion Matrix")
            interp = ClassificationInterpretation.from_learner(self.model)
            interp.plot_confusion_matrix(figsize=(16, 16), normalize=True)
            plt.savefig(f"{self.model_path}/confusion_matrix.png")
            st.image(Image.open(f"{self.model_path}/confusion_matrix.png"), use_container_width=False)


        
        



class LogCallback(Callback):
    # def before_fit(self):
    #     st.session_state.training_out = st.empty()
    #     st.session_state.progress =  st.progress(0.0)

    def after_batch(self):
        if self.training:
            batch = self.iter
            loss = self.loss.item()
            st.session_state.training_out.code(f"Batch {batch}: Loss {loss:.4f}")
            st.session_state.progress.progress(batch/ self.n_iter)


    def after_epoch(self):
        st.session_state.progress.progress(1.0)
        st.session_state.training_out.code(f"Epoch {self.epoch} complete!")
    
    # def before_epoch(self):
    #     fig, ax = plt.subplots(figsize=(3, 3))
    #     self.recorder.plot_loss(ax=ax)
    #     st.session_state.training_table.pyplot(fig, use_container_width=False)

# class StreamlitLogger(io.StringIO):
#     def write(self, message):
#         # st.session_state.training_out.text(message.strip())
#         st.code(message.strip())
