# from tensorflow.python.client import device_lib

# device_lib.list_local_devices()
# print(5)
# exit(0)


import runTf


trim = False

catagory = "stiff/stiffC/top"
# catagory = "stiff/stiffC/bottom"
# catagory = "stiff/stiffT/top"
# catagory = "stiff/stiffT/bottom"

# catagory = "features/aveCB/top"
# catagory = "features/aveCB/bottom"
# catagory = "features/aveCF/top"
# catagory = "features/aveCF/bottom"
# catagory = "features/aveTB/top"
# catagory = "features/aveTB/bottom"
# catagory = "features/aveTF/top"
# catagory = "features/aveTF/bottom"
# runTf.train.moment(catagory, trim, 2)
# runTf.train.point(catagory, trim)
runTf.train.voxel(catagory, trim)




# catagory = ["stiff/stiffC/top", "stiff/stiffC/bottom", "stiff/stiffT/top", "stiff/stiffT/bottom",
			# "features/aveCB/top", "features/aveCB/bottom", "features/aveCF/top", "features/aveCF/bottom",
			# "features/aveTB/top", "features/aveTB/bottom", "features/aveTF/top", "features/aveTF/bottom"]
			

	
# for cat in catagory:
	# runTf.group.continous(cat, trim)

	
	
# catagory = ["jump/jumpC", "jump/jumpT"]

# for cat in catagory:
	# runTf.group.jump(cat, trim)

