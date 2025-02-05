from fastai.vision.all import *
import streamlit as st
import os

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

        #sys.stdout = StreamlitLogger()
        self.save_path = f".tmp/{st.session_state.username}"
        try:
            os.mkdir(self.save_path)
        except:
            pass

    def train_model(self, architecture_name, transform_type, zipFile, num_epochs, num_batches, callbacks) -> None:
        st.session_state.training_out = st.empty()
        st.session_state.training_table = st.empty()
        st.session_state.progress =  st.progress(0)
        # tmp_path = "/".join(zipFile.split('/')[:-1])
        # if zipfile.is_zipfile(zipFile.name):
        #     with zipfile.ZipFile(zipFile.name, 'r') as zip_ref:
        #         zip_ref.extractall(tmp_path)
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
            self.model = vision_learner(dls, self.architectures[architecture_name], metrics=F1Score(average='macro'), path=self.save_path)
            self.model.fine_tune(num_epochs, cbs=[eval(callbacks), LogCallback])

            weights_path = f"{architecture_name}_{transform_type}.pkl"
            self.model.export(weights_path)
            st.session_state.uploaded_file = f"{self.save_path}/{weights_path}"

            st.success(f"Training completed! Model save as {weights_path}.")

            self.__plot_loss()
        except Exception as e:
            st.error(f"Error in training: {str(e)}")

    def __plot_loss(self):
        # Exemplo de uso do layout com colunas
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📉 Training & Validation Loss")
            fig, ax = plt.subplots(figsize=(3, 3))
            self.model.recorder.plot_loss(ax=ax)
            st.pyplot(fig, use_container_width=False)  # Use container width for better layout

        with col2:
            st.subheader("🔢 Confusion Matrix")
            interp = ClassificationInterpretation.from_learner(self.model)
            fig, ax = plt.subplots(figsize=(4, 4))
            interp.plot_confusion_matrix()
            st.pyplot(fig, use_container_width=False)  # Use container width for better layout


class LogCallback(Callback):
    def after_batch(self):
        if self.training:
            batch = self.iter
            loss = self.loss.item()
            st.session_state.training_out.code(f"Batch {batch}: Loss {loss:.4f}")
            st.session_state.progress.progress(batch/self.n_iter)

    def after_epoch(self):
        st.session_state.progress.progress(1.0)
        st.session_state.training_out.code(f"Epoch {self.epoch+1} complete!")

class StreamlitLogger(io.StringIO):
    def write(self, message):
        # st.session_state.training_out.text(message.strip())
        # st.write(message.strip())
        pass