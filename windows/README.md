
# Using
## Only test
- 1.	Put the 68.model and 68_half.model to this directory 
- 2. cd $build  
-	- 3000fps-mtcnn video ====> using mtcnn to detect face and 3000fps alignment
-	- 3000fps-mtcnn video half ====> using accelerated version mtcnn to detect face and 3000fps alignment

## Train
	build a new project and add the source file .head file ,configure Caffe and Opencv,prepare datalist into data/68/ 
	
	directory and build  
	
	run 3000fps-mtcnn prepare to generator train.txt and test.txt in data/68/
	
	run 3000fps-mtcnn train to generator model file in model directory;
	
	run 3000fps-mtcnn video to test video result;
	

## Something Else
   in order to accelerate the detect speed, you can decrease the bbox generator in pNet, Increase the threshold 
is useful .
