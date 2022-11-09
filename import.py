import glob
from core.data import StableDiffusionData

def main():
    for diffusers_directory in glob.glob("import/diffusers/*"):
        data = StableDiffusionData()
        data.load_diffusers(diffusers_directory)
        print(data)
        print()
        data.save_native()

    for checkpoint_filename in glob.glob("import/checkpoints/*.ckpt"):
        data = StableDiffusionData()
        data.load_checkpoint(checkpoint_filename)
        print(data)
        print()
        data.save_native()

if __name__ == "__main__":
    main()

