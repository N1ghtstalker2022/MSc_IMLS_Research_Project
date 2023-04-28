# MSc IMLS Research Project
# **Spherical  Convolutions for Neural Compression of 360-degree Videos**

This  project focus on the synergy of two current learning methods for video. The  first is neural video compression[1], which improves traditional one certain  steps. Traditional compression captures motion between consecutive frames and  performs linear transform and quantisation methods over the residual  information in each frame to be encoded. Differently, neural video  compression uses methods such as convolutional neural networks (CNNs) to  capture motion and encoder-decoder architectures to propose to compress  residual data. The second is spherical convolutional neural networks (CNNs)  for 360-degree videos. The current compression of such 360-degree videos  intensively relies on 2D planar projection and suffers from its distortion  (e.g., poles are more sampled than equator). Therefore, neural compression  using CNNs, which focus on 2D convolution over planar data, is damaged when  applied to 360 videos. Some authors[3] have evaluated the benefit of using  spherical convolutions in learning tasks for 360 images.          

The project is experiments-oriented, and the student will focus on  implementing neural compression methods but changing the CNN steps by ones  using spherical convolutions.         

 [1] Eirikur Agustsson, David Minnen, Nick Johnston, Johannes Balle, Sung  Jin Hwang, and George Toderici. 2020. Scale-Space Flow for End-to-End  Opti-mized Video Compression. In 2020 IEEE/CVF Conference on Computer Vision  and Pattern Recognition (CVPR) (Seattle, WA, USA). IEEE, 8500–8509     

[2] Guo Lu, Wanli Ouyang, Dong Xu, Xiaoyun Zhang, Chunlei Cai, and Zhiyong  Gao. 2019. DVC: An End-To-End Deep Video Compression Framework.  11006–11015     

[3] Navid Mahmoudian Bidgoli, Roberto G. de A. Azevedo, Thomas Maugey,  Aline Roumy, and Pascal Frossard. 2021. OSLO: On-the-Sphere Learning for  Omnidirectional Images and Its Application to 360-Degree Image Compression.  (2021). arXiv:2107.09179     



Bigoli at. al work proposes spherical convolution but only uses it for image compression. On the other hand, the current neural video compression focuses on 2D video using 2D convolutions. So, the idea is to replace 2D convolution steps in current neural video compression with spherical convolutions for 360 videos.

The suggested steps are: to implement spherical convolution; set up one reference method neural video compression (you can use open code as [1]); replace 2D convolution steps in this compression for the spherical one; then compare results with compression metrics against the reference compression.
