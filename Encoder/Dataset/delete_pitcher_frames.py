import os

# Path to folder in which delete frames of pitcher
path_to_dir = "/media/sealab-ws/Hard Disk/CORSMAL challenge/train_patches/dataset_pulito"
id_list = [
    "000002_0.png",
    "000002_1.png",
    "000002_2.png",
    "000002_3.png",
    "000002_4.png",
    "000004_3.png",
    "000009_3.png",
    "000009_4.png",
    "000010_2.png",
    "000010_3.png",
    "000010_4.png",
    "000013_0.png",
    "000013_1.png",
    "000013_2.png",
    "000013_3.png",
    "000013_4.png",
    "000015_4.png",
    "000022_0.png",
    "000022_1.png",
    "000022_2.png",
    "000022_3.png",
    "000022_4.png",
    "000023_3.png",
    "000023_4.png",
    "000026_0.png",
    "000026_1.png",
    "000026_2.png",
    "000026_3.png",
    "000026_4.png",
    "000035_2.png",
    "000035_3.png",
    "000035_4.png",
    "000053_0.png",
    "000053_1.png",
    "000053_2.png",
    "000053_3.png",
    "000053_4.png",
    "000068_2.png",
    "000068_3.png",
    "000068_4.png",
    "000069_4.png",
    "000070_0.png",
    "000070_1.png",
    "000070_2.png",
    "000070_3.png",
    "000070_4.png",
    "000086_0.png",
    "000086_1.png",
    "000086_2.png",
    "000086_3.png",
    "000086_4.png",
    "000092_4.png",
    "000130_0.png",
    "000130_1.png",
    "000130_2.png",
    "000130_3.png",
    "000130_4.png",
    "000131_4.png",
    "000137_0.png",
    "000137_1.png",
    "000137_2.png",
    "000137_3.png",
    "000137_4.png",
    "000138_0.png",
    "000138_1.png",
    "000138_2.png",
    "000138_3.png",
    "000138_4.png",
    "000141_0.png",
    "000141_1.png",
    "000141_2.png",
    "000141_3.png",
    "000141_4.png",
    "000182_0.png",
    "000182_1.png",
    "000182_2.png",
    "000182_3.png",
    "000199_1.png",
    "000199_3.png",
    "000199_4.png",
    "000203_3.png",
    "000203_4.png",
    "000219_4.png",
    "000221_3.png",
    "000221_4.png",
    "000229_0.png",
    "000229_1.png",
    "000229_2.png",
    "000229_3.png",
    "000229_4.png",
    "000237_0.png",
    "000237_1.png",
    "000237_2.png",
    "000237_3.png",
    "000237_4.png",
    "000238_0.png",
    "000238_1.png",
    "000238_2.png",
    "000238_3.png",
    "000238_4.png",
    "000239_0.png",
    "000239_1.png",
    "000239_2.png",
    "000239_3.png",
    "000239_4.png",
    "000246_0.png",
    "000246_1.png",
    "000246_2.png",
    "000246_3.png",
    "000246_4.png",
    "000247_1.png",
    "000247_2.png",
    "000247_3.png",
    "000249_2.png",
    "000249_3.png",
    "000249_4.png",
    "000268_0.png",
    "000268_1.png",
    "000268_2.png",
    "000268_3.png",
    "000268_4.png",
    "000273_0.png",
    "000273_1.png",
    "000273_2.png",
    "000273_3.png",
    "000273_4.png",
    "000279_3.png",
    "000279_4.png",
    "000288_0.png",
    "000288_1.png",
    "000288_2.png",
    "000288_3.png",
    "000288_4.png",
    "000290_4.png",
    "000297_3.png",
    "000297_4.png",
    "000299_0.png",
    "000299_1.png",
    "000299_2.png",
    "000299_3.png",
    "000299_4.png",
    "000308_0.png",
    "000308_1.png",
    "000308_2.png",
    "000308_3.png",
    "000308_4.png",
    "000315_3.png",
    "000315_4.png",
    "000319_0.png",
    "000319_1.png",
    "000319_2.png",
    "000319_3.png",
    "000319_4.png",
    "000324_0.png",
    "000324_1.png",
    "000324_2.png",
    "000324_3.png",
    "000324_4.png",
    "000370_0.png",
    "000370_1.png",
    "000370_2.png",
    "000370_3.png",
    "000370_4.png",
    "000372_0.png",
    "000372_1.png",
    "000372_2.png",
    "000372_3.png",
    "000372_4.png",
    "000375_0.png",
    "000375_1.png",
    "000375_2.png",
    "000375_3.png",
    "000375_4.png",
    "000382_0.png",
    "000382_1.png",
    "000382_2.png",
    "000382_3.png",
    "000382_4.png",
    "000384_1.png",
    "000384_2.png",
    "000384_4.png",
    "000412_1.png",
    "000412_2.png",
    "000412_3.png",
    "000412_4.png",
    "000421_0.png",
    "000421_1.png",
    "000421_2.png",
    "000421_3.png",
    "000425_0.png",
    "000425_1.png",
    "000425_2.png",
    "000425_3.png",
    "000425_4.png",
    "000429_0.png",
    "000429_1.png",
    "000429_2.png",
    "000429_3.png",
    "000429_4.png",
    "000431_4.png",
    "000435_0.png",
    "000435_1.png",
    "000435_2.png",
    "000435_3.png",
    "000435_4.png",
    "000443_4.png",
    "000463_0.png",
    "000463_1.png",
    "000463_2.png",
    "000463_3.png",
    "000463_4.png",
    "000469_0.png",
    "000469_1.png",
    "000469_2.png",
    "000479_0.png",
    "000479_1.png",
    "000479_2.png",
    "000479_3.png",
    "000479_4.png",
    "000490_1.png",
    "000490_2.png",
    "000490_3.png",
    "000490_4.png",
    "000494_0.png",
    "000494_1.png",
    "000494_2.png",
    "000494_3.png",
    "000494_4.png",
    "000497_0.png",
    "000497_1.png",
    "000502_1.png",
    "000502_2.png",
    "000502_3.png",
    "000502_4.png",
    "000505_0.png",
    "000505_1.png",
    "000505_2.png",
    "000505_3.png",
    "000505_4.png",
    "000538_0.png",
    "000538_1.png",
    "000538_2.png",
    "000538_3.png",
    "000538_4.png",
    "000559_0.png",
    "000559_1.png",
    "000559_2.png",
    "000559_3.png",
    "000559_4.png",
    "000566_3.png",
    "000566_4.png",
    "000573_0.png",
    "000573_1.png",
    "000573_2.png",
    "000573_3.png",
    "000573_4.png",
    "000602_4.png",
    "000605_0.png",
    "000605_1.png",
    "000605_2.png",
    "000605_3.png",
    "000605_4.png",
    "000606_0.png",
    "000606_1.png",
    "000606_2.png",
    "000606_3.png",
    "000606_4.png",
    "000607_1.png",
    "000607_3.png",
    "000607_4.png",
    "000667_4.png"]
folders = ["rgb", "mask", "depth"]
print(len(id_list)/3408)
for id in id_list:
    for f in folders:
        path_to_folder = os.path.join(path_to_dir, f)
        os.remove(os.path.join(path_to_folder, id))
