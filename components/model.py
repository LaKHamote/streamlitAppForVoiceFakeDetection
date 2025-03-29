from fastai.vision.all import *
import streamlit as st
import streamlit_ext as ste
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


    def train_model(self, architecture_name, transform_type, selected_speakers, num_epochs, num_batches, callbacks) -> None:
        self.__init_logs()

        if architecture_name not in self.architectures:
            st.warning("Architecture not suported.")

        if transform_type in self.transforms:
            transform = self.transforms[transform_type]
        else:
            st.warning("Transformation not suported.")

        try:
            dataset_path = "components/VoCoderRecognition/dataset/generated/spec/0/"
            tmp_paths=[f"{dataset_path}{spk}" for spk in selected_speakers]
            
            dls = ImageDataLoaders.from_path_func(
                path=".",
                fnames=[f for path in tmp_paths for f in get_image_files(Path(path))],
                label_func=label_func,
                bs=num_batches,
                valid_pct=0.3,
                item_tfms=transform
            )
            st.session_state.dataset_info.info(f"✅ Selected {len(dls.train.dataset)+len(dls.valid.dataset)} images. Training will start soon!")
            time.sleep(3)

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

        try:
            self.model_path = f"{self.save_path}/{architecture_name}_{transform_type}"
            os.makedirs(self.model_path, exist_ok=True)
            self.model = vision_learner(dls, self.architectures[architecture_name], metrics=F1Score(average='macro'), path=self.model_path)
            self.model.fine_tune(num_epochs, cbs=[eval(callbacks), LogCallback, GraphCallback])

            self.model.export("model.pkl")
            st.session_state.trained_model = f"{self.model_path}/model.pkl"

            st.success("✅ Training completed! Model saved in cache. Please download it before finishing your session.")
            self.__empty_logs()
            self.history_data = pd.read_csv(f"{self.model_path}/history.csv")
        
            st.table(self.history_data)

            self.__plot_loss()

            model_buffer = io.BytesIO()
            with open(f"{self.model_path}/model.pkl", "rb") as file:
                model_buffer.write(file.read())  
            model_buffer.seek(0)
            st.download_button(
                label="🚀 Download Model", 
                data=model_buffer, 
                file_name="model.pkl", 
                mime="application/octet-stream"
            )
        except Exception as e:
            st.error(f"Error in training: {str(e)}")
    
    def __init_logs(self):
        st.session_state.dataset_info = st.empty()
        st.session_state.training_out = st.empty()
        st.session_state.progress =  st.progress(0.0)
        st.session_state.graph = st.empty()
    
    def __empty_logs(self):
        st.session_state.dataset_info.empty()
        st.session_state.training_out.empty()
        st.session_state.progress.empty()
        st.session_state.graph.empty()

    def __plot_loss(self):

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📉 Training & Validation Loss")
            fig, ax = plt.subplots(figsize=(3, 3))
            self.model.recorder.plot_loss(ax=ax)
            st.pyplot(fig, use_container_width=False)


            csv_buffer =  io.BytesIO()
            self.history_data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            # Display loss values

            st.download_button(
                label="📉 Download Training & Validation Loss", 
                data=csv_buffer, 
                file_name="loss_values.csv", 
                mime="text/csv"
            )

        with col2:
            st.subheader("🔢 Confusion Matrix")
            interp = ClassificationInterpretation.from_learner(self.model)
            img_buffer = io.BytesIO()
            interp.plot_confusion_matrix(figsize=(16, 16), normalize=True)
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            st.image(img_buffer, use_container_width=False)

            st.download_button(
                label="📥 Download Confusion Matrix", 
                data=img_buffer, 
                file_name="confusion_matrix.png", 
                mime="image/png"
            )

def label_func(f): 
    return f.parent.name

class LogCallback(Callback):
    def after_batch(self):
        if self.training:
            batch = self.iter
            loss = self.loss.item()
            st.session_state.dataset_info.empty()
            st.session_state.training_out.code(f"Epoch {self.epoch}: \nBatch {batch}: Loss {loss:.4f}")
            st.session_state.progress.progress(batch/ self.n_iter)

    def after_epoch(self):
        st.session_state.progress.progress(1.0)
        st.session_state.training_out.code(f"Epoch {self.epoch} complete!")

class GraphCallback(Callback):
    def after_epoch(self):
        fig, ax = plt.subplots(figsize=(3, 3))
        self.recorder.plot_loss(ax=ax)
        st.session_state.graph.pyplot(fig, use_container_width=False)
