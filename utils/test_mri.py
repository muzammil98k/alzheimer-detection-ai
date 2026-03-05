import nibabel as nib

path = "data/raw/disc1/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.img"

img = nib.load(path)
data = img.get_fdata()

print("Shape:", data.shape)
