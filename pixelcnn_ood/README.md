train_pixelcnn.py
	
	To train the PixelCNN++ model and evaluate LL. Also can train the background model for LRat, and evaluate LRat.


stirring_ll.py

	To evaluate stirring LL introduced in the paper (with and without conditional correction).


shaking_ll.py
	
	To evaluate shaking LL introduced in the paper (with and without conditional correction).


probs_ic.ipynb

	To evaluate IC performance with PNG, JPEG, or FLIF.



Dependencies


	network.py
	
		Contains the exact same code as tfp.distributions.PixelCNN, but additionally has provisions to drop blocks or short-cut connections.

	
	dataset_utils.py

		Load and preprocess datasets as mentioned in Appendix B. Random seeds used are included.

	
	utils.py

		Containes utilities to checkpoint model training.


	transform_utils.py

		Contains functions to perform stirring or shaking on images.


	metric_utils.py
	
		Contains function to compute outlier detection performance given likelihoods.



Requirements


tensorflow-gpu==2.6.0

tensorflow-probability==0.14.1

tensorflow-datasets==4.5.2

numpy==1.19.5

scikit-learn==1.1.1

opencv-python==4.6.0.66

scikit-image==0.19.3
