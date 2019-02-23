the file src/predict_3pose.py performs the whole nine-yards of the work outlined from the research paper.
Given different input flags the file will train the algorithm on different parameters and datasets (gt or stacked hourglass)
and then will run the tests for that trained algorithm. 

Two things that are important to know... When training the algorithm the camera coordinates must be know 
but this is not important for testing the algorithm. The second thing to know is that the 
model takes 2d coordinates as an input. This 2d coordinates that are generated for predicted 
stacked hour glass 2d coordinates are created with the following function call from predict_3dpose.py

train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, FLAGS.data_dir, rcams )

linear_model.py is where the deep learnining model is defined using tensorflow. 

