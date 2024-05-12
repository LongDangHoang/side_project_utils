import boto3
import wandb

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback

from pathlib import Path


class S3SyncCallback(Callback):
    """
    Synchronise checkpoint folder with bucket
    
    Initialises with the checkpoint folder. Path from current working directory to checkpoint folder will be used 
    as key to s3.
    """
    def __init__(
        self, 
        save_local_dir: Path, 
        load_local_dir: Path=None, 
        root_path: Path=Path('.'),
        every_n_epochs: int=1,
    ) -> None:
        self.s3 = boto3.resource("s3")
        self.bucket_name = 'longdang-deep-learning-personal-projects'
        self.bucket = self.s3.Bucket(self.bucket_name)
        
        assert root_path.exists()
        self.root_path = root_path.resolve()

        self.save_local_dir = save_local_dir
        self.load_local_dir = load_local_dir if load_local_dir else save_local_dir
        self.save_s3_key = self.__get_s3_key(self.save_local_dir)
        self.load_s3_key = self.__get_s3_key(self.load_local_dir)
        
        self.every_n_epochs = every_n_epochs
        self.epoch_counter_state = 0
        
        print("Initialised S3 sync. Saving to:", self.save_s3_key, "and loading from:", self.load_s3_key)
    
    def __get_s3_key(self, path: Path) -> str:
        return str(path.absolute().relative_to(self.root_path))
        
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.epoch_counter_state = (
            trainer.current_epoch
            if hasattr(trainer, "current_epoch")
            else self.epoch_counter_state + 1
        )
        if self.epoch_counter_state % self.every_n_epochs == 0:
            self.upload_files_to_s3()
            
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.upload_files_to_s3()
        
    def download_files_from_s3(self):
        for file in self.bucket.objects.filter(Prefix=self.load_s3_key):
            filename = file.key.split("/")[-1]
            self.bucket.download_file(file.key, self.load_local_dir / filename)
            
    def upload_files_to_s3(self):
        self.delete_folder_on_s3()
        self.clean_save_local_dir()
        
        for file in self.save_local_dir.rglob("*"):
            key = self.__get_s3_key(file)
            self.bucket.upload_file(file, key)
            
    def delete_folder_on_s3(self):
        for file in self.bucket.objects.filter(Prefix=self.save_s3_key):
            self.s3.Object(self.bucket_name, file.key).delete()
    
    def clean_save_local_dir(self):   
        # check if last_v1 is there, make it last if needed
        if (self.save_local_dir / "last-v1.ckpt").exists():
            (self.save_local_dir / "last-v1.ckpt").replace(
                self.save_local_dir / "last.ckpt"
            )    
            
    def download_filename(self, filename: str):
        assert isinstance(filename, str), "Please provide filename as string, not a path object"
        self.bucket.download_file(self.load_s3_key + "/" + filename, self.load_local_dir / filename)
        
    def upload_filename(self, filename: str):
        assert isinstance(filename, str), "Please provide filename as string, not a path object"
        self.bucket.upload_file(self.save_local_dir / filename, self.save_s3_key + "/" + filename)
