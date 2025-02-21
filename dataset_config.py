from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="/fp/projects01/ec35/data/IN2012/",
                       extra="/fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/dinov2/dinov2/data/extra")
    dataset.dump_extra()
