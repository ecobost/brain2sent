# Written by: Erick Cobos T
# Date: 15-Sep-2016
""" Small script to make an html file to showcase sample descriptions. 

I print 6 images per row in an html table for a total of 89 rows (534 images).
I ignore the last 6 images because I couldn't possibly predict anything good for
them as their effect appears in future BOLD data which I don't have.

I just print the inside of the table and then copy and paste it in the html file.
"""

fps = 15 # frames per second. Number of images for every second

# Read file with sentence descriptions
sents_file = open('test_sents_pred.txt', 'r')

# Over all rows
for row in range(89):

	# print the image row
	print('<tr>')
	for col in range(6):
		img_number = row*(6*fps) + col*fps + round(fps/2)
		#os.system('cp test_images/img_{}.png test_images2/'.format(img_number))
		img_src = 'data/test_images/img_{}.png'.format(img_number)
		print('\t<td><img src="' + img_src + '" style="width:150px;height:150px;"></td>')
	print('</tr>')
	
	# print the description row
	print('<tr>')
	for col in range(6):
		description = sents_file.readline().rstrip('\n')
		print('\t<td>' + description + '</td>')
	print('</tr>')
