# pytorch-unet
A factory of U-Net, could easily change backbone like ResNet or ResNeXt.

* As we known, U-Net has the 'U' shape, left half is encoder, right half is decoder, and the most important is that encoder should produce shortcuts for decoder to concatenate.
* So, I make an abstraction, define a factory of U-Net, then we could custome encoder and decoder as we want but just has to follow some rules.
* I had already finish U-Net-ResNet18/34/50/101/150 and U-Net-ResNeXt50(32x4d). Have fun!
