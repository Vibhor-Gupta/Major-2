#1=17
#2=11
#3=257
#4=167
#5=59
#6=54
#7=185
# from sklearn.cross_validation import train_test_split
import random

label =	{	
			"1.jpg" :3,
			"2.jpg" :3,
			"3.jpg" :7,
			"4.jpg" :4,
			"5.jpg" :7,
			"6.jpg" :3,
			"7.jpg" :3,
			"8.jpg" :7,
			"9.jpg" :4,
			"10.jpg" :7,
			"11.jpg" :3,
			"12.jpg" :3,
			"13.jpg" :7,
			"14.jpg" :4,
			"15.jpg" :7,
			"16.jpg" :3,
			"17.jpg" :3,
			"18.jpg" :7,
			"19.jpg" :4,
			"20.jpg" :7,
			"21.jpg" :3,
			"22.jpg" :3,
			"23.jpg" :7,
			"24.jpg" :4,
			"25.jpg" :7,
			"26.jpg" :3,
			"27.jpg" :3,
			"28.jpg" :7,
			"29.jpg" :4,
			"30.jpg" :7,
			"31.jpg" :4,
			"32.jpg" :7,
			"33.jpg" :6,
			"34.jpg" :4,
			"35.jpg" :7,
			"36.jpg" :3,
			"37.jpg" :6,
			"38.jpg" :4,
			"39.jpg" :7,
			"40.jpg" :3,
			"41.jpg" :6,
			"42.jpg" :4,
			"43.jpg" :7,
			"44.jpg" :3,
			"45.jpg" :6,
			"46.jpg" :4,
			"47.jpg" :7,
			"48.jpg" :3,
			"49.jpg" :6,
			"50.jpg" :4,
			"51.jpg" :7,
			"52.jpg" :3,
			"53.jpg" :6,
			"54.jpg" :7,
			"55.jpg" :3,
			"56.jpg" :6,
			"57.jpg" :4,
			"58.jpg" :7,
			"59.jpg" :6,
			"60.jpg" :4,
			"61.jpg" :7,
			"62.jpg" :3,
			"63.jpg" :6,
			"64.jpg" :4,
			"65.jpg" :7,
			"66.jpg" :3,
			"67.jpg" :6,
			"68.jpg" :4,
			"69.jpg" :7,
			"70.jpg" :3,
			"71.jpg" :6,
			"72.jpg" :4,
			"73.jpg" :7,
			"74.jpg" :3,
			"75.jpg" :6,
			"76.jpg" :4,
			"77.jpg" :7,
			"78.jpg" :3,
			"79.jpg" :3,
			"80.jpg" :6,
			"81.jpg" :4,
			"82.jpg" :6,
			"83.jpg" :4,
			"84.jpg" :3,
			"85.jpg" :6,
			"86.jpg" :4,
			"87.jpg" :3,
			"88.jpg" :6,
			"89.jpg" :4,
			"90.jpg" :3,
			"91.jpg" :6,
			"92.jpg" :4,
			"93.jpg" :3,
			"94.jpg" :6,
			"95.jpg" :4,
			"96.jpg" :3,
			"97.jpg" :3,
			"98.jpg" :3,
			"99.jpg" :6,
			"100.jpg" :4,
			"101.jpg" :3,
			"102.jpg" :6,
			"103.jpg" :4,
			"104.jpg" :3,
			"105.jpg" :6,
			"106.jpg" :4,
			"107.jpg" :3,
			"108.jpg" :6,
			"109.jpg" :4,
			"110.jpg" :3,
			"111.jpg" :6,
			"112.jpg" :4,
			"113.jpg" :6,
			"114.jpg" :4,
			"115.jpg" :3,
			"116.jpg" :3,
			"117.jpg" :6,
			"118.jpg" :3,
			"119.jpg" :3,
			"120.jpg" :3,
			"121.jpg" :6,
			"122.jpg" :4,
			"123.jpg" :7,
			"124.jpg" :3,
			"125.jpg" :3,
			"126.jpg" :3,
			"127.jpg" :6,
			"128.jpg" :4,
			"129.jpg" :7,
			"130.jpg" :3,
			"131.jpg" :3,
			"132.jpg" :3,
			"133.jpg" :6,
			"134.jpg" :4,
			"135.jpg" :7,
			"136.jpg" :3,
			"137.jpg" :3,
			"138.jpg" :3,
			"139.jpg" :6,
			"140.jpg" :4,
			"141.jpg" :7,
			"142.jpg" :3,
			"143.jpg" :3,
			"144.jpg" :3,
			"145.jpg" :6,
			"146.jpg" :4,
			"147.jpg" :7,
			"148.jpg" :3,
			"149.jpg" :4,
			"150.jpg" :7,
			"151.jpg" :3,
			"152.jpg" :3,
			"153.jpg" :6,
			"154.jpg" :3,
			"155.jpg" :6,
			"156.jpg" :4,
			"157.jpg" :3,
			"158.jpg" :3,
			"159.jpg" :3,
			"160.jpg" :6,
			"161.jpg" :4,
			"162.jpg" :3,
			"163.jpg" :3,
			"164.jpg" :3,
			"165.jpg" :6,
			"166.jpg" :4,
			"167.jpg" :3,
			"168.jpg" :3,
			"169.jpg" :3,
			"170.jpg" :6,
			"171.jpg" :4,
			"172.jpg" :3,
			"173.jpg" :3,
			"174.jpg" :3,
			"175.jpg" :6,
			"176.jpg" :4,
			"177.jpg" :3,
			"178.jpg" :3,
			"179.jpg" :4,
			"180.jpg" :3,
			"181.jpg" :3,
			"182.jpg" :3,
			"183.jpg" :3,
			"184.jpg" :4,
			"185.jpg" :3,
			"186.jpg" :4,
			"187.jpg" :3,
			"188.jpg" :3,
			"189.jpg" :3,
			"190.jpg" :3,
			"191.jpg" :4,
			"192.jpg" :3,
			"193.jpg" :3,
			"194.jpg" :3,
			"195.jpg" :4,
			"196.jpg" :3,
			"197.jpg" :3,
			"198.jpg" :3,
			"199.jpg" :4,
			"200.jpg" :3,
			"201.jpg" :3,
			"202.jpg" :3,
			"203.jpg" :4,
			"204.jpg" :3,
			"205.jpg" :3,
			"206.jpg" :3,
			"207.jpg" :3,
			"208.jpg" :3,
			"209.jpg" :4,
			"210.jpg" :3,
			"211.jpg" :4,
			"212.jpg" :3,
			"213.jpg" :3,
			"214.jpg" :3,
			"215.jpg" :4,
			"216.jpg" :3,
			"217.jpg" :3,
			"218.jpg" :4,
			"219.jpg" :2,
			"220.jpg" :3,
			"221.jpg" :3,
			"222.jpg" :4,
			"223.jpg" :2,
			"224.jpg" :3,
			"225.jpg" :4,
			"226.jpg" :2,
			"227.jpg" :2,
			"228.jpg" :2,
			"229.jpg" :3,
			"230.jpg" :4,
			"231.jpg" :2,
			"232.jpg" :3,
			"233.jpg" :4,
			"234.jpg" :2,
			"235.jpg" :3,
			"236.jpg" :3,
			"237.jpg" :4,
			"238.jpg" :2,
			"239.jpg" :1,
			"240.jpg" :3,
			"241.jpg" :3,
			"242.jpg" :4,
			"243.jpg" :2,
			"244.jpg" :1,
			"245.jpg" :3,
			"246.jpg" :3,
			"247.jpg" :3,
			"248.jpg" :4,
			"249.jpg" :2,
			"250.jpg" :1,
	    	 "251.jpg":3,
			 "252.jpg":3,
			 "253.jpg":3,
			 "254.jpg":4,
			 "255.jpg":2,
			 "256.jpg":3,
			 "257.jpg":1,
			 "258.jpg":3,
			 "259.jpg":3,
			 "260.jpg":3,
			 "261.jpg":4,
			 "262.jpg":1,
			 "263.jpg":3,
			 "264.jpg":4,
			 "265.jpg":3,
			 "266.jpg":3,
			 "267.jpg":3,
			 "268.jpg":4,
			 "269.jpg":3,
			 "270.jpg":3,
			 "271.jpg":4,
			 "272.jpg":3,
			 "273.jpg":3,
			 "274.jpg":3,
			 "275.jpg":3,
			 "276.jpg":4,
			 "277.jpg":1,
			 "278.jpg":3,
			 "279.jpg":3,
			 "280.jpg":4,
			 "281.jpg":1,
			 "282.jpg":3,
			 "283.jpg":3,
			 "284.jpg":1,
			 "285.jpg":3,
			 "286.jpg":1,
			 "287.jpg":3,
			 "288.jpg":3,
			 "289.jpg":1,
			 "290.jpg":3,
			 "291.jpg":3,
			 "292.jpg":4,
			 "293.jpg":1,
			 "294.jpg":3,
			 "295.jpg":4,
			 "296.jpg":3,
			 "297.jpg":3,
			 "298.jpg":3,
			 "299.jpg":3,
			 "300.jpg":4,
			 "301.jpg":3,
			 "302.jpg":3,
			 "303.jpg":3,
			 "304.jpg":4,
			 "305.jpg":1,
			 "306.jpg":3,
			 "307.jpg":3,
			 "308.jpg":3,
			 "309.jpg":4,
			 "310.jpg":1,
			 "311.jpg":3,
			 "312.jpg":3,
			 "313.jpg":4,
			 "314.jpg":1,
			 "315.jpg":3,
			 "316.jpg":3,
			 "317.jpg":1,
			 "318.jpg":3,
			 "319.jpg":3,
			 "320.jpg":1,
			 "321.jpg":3,
			 "322.jpg":1,
			 "323.jpg":3,
			 "324.jpg":3,
			 "325.jpg":3,
			 "326.jpg":3,
			 "327.jpg":3,
			 "328.jpg":3,
			 "329.jpg":3,
			 "330.jpg":3,
			 "331.jpg":3,
			 "332.jpg":7,
			 "333.jpg":7,
			 "334.jpg":7,
			 "335.jpg":7,
			 "336.jpg":5,
			 "337.jpg":7,
			 "338.jpg":5,
			 "339.jpg":7,
			 "340.jpg":5,
			 "341.jpg":5,
			 "342.jpg":5,
			 "343.jpg":5,
			 "344.jpg":5,
			 "345.jpg":5,
			 "346.jpg":5,
			 "347.jpg":5,
			 "348.jpg":5,
			 "349.jpg":5,
			 "350.jpg":3,
			 "351.jpg":3,
			 "352.jpg":5,
			 "353.jpg":3,
			 "354.jpg":5,
			 "355.jpg":3,
			 "356.jpg":5,
			 "357.jpg":3,
			 "358.jpg":5,
			 "359.jpg":3,
			 "360.jpg":5,
			 "361.jpg":5,
			 "362.jpg":3,
			 "363.jpg":5,
			 "364.jpg":3,
			 "365.jpg":5,
			 "366.jpg":3,
			 "367.jpg":5,
			 "368.jpg":3,
			 "369.jpg":5,
			 "370.jpg":3,
			 "371.jpg":5,
			 "372.jpg":5,
			 "373.jpg":3,
			 "374.jpg":3,
			 "375.jpg":3,
			 "376.jpg":3,
			 "377.jpg":3,
			 "378.jpg":3,
			 "379.jpg":4,
			 "380.jpg":4,
			 "381.jpg":4,
			 "382.jpg":4,
			 "383.jpg":4,
			 "384.jpg":4,
			 "385.jpg":4,
			 "386.jpg":4,
			 "387.jpg":4,
			 "388.jpg":4,
			 "389.jpg":4,
			 "390.jpg":4,
			 "391.jpg":4,
			 "392.jpg":4,
			 "393.jpg":4,
			 "394.jpg":4,
			 "395.jpg":4,
			 "396.jpg":4,
			 "397.jpg":4,
			 "398.jpg":4,
			 "399.jpg":4,
			 "400.jpg":4,
			 "401.jpg":4,
			 "402.jpg":4,
			 "403.jpg":4,
			 "404.jpg":4,
			 "405.jpg":4,
			 "406.jpg":4,
			 "407.jpg":4,
			 "408.jpg":4,
			 "409.jpg":4,
			 "410.jpg":4,
			 "411.jpg":4,
			 "412.jpg":4,
			 "413.jpg":4,
			 "414.jpg":4,
			 "415.jpg":4,
			 "416.jpg":4,
			 "417.jpg":4,
			 "418.jpg":4,
			 "419.jpg":4,
			 "420.jpg":4,
			 "421.jpg":4,
			 "422.jpg":4,
			 "423.jpg":4,
			 "424.jpg":4,
			 "425.jpg":4,
			 "426.jpg":4,
			 "427.jpg":4,
			 "428.jpg":4,
			 "429.jpg":4,
			 "430.jpg":4,
			 "431.jpg":4,
			 "432.jpg":4,
			 "433.jpg":4,
			 "434.jpg":4,
			 "435.jpg":4,
			 "436.jpg":4,
			 "437.jpg":4,
			 "438.jpg":4,
			 "439.jpg":4,
			 "440.jpg":4,
			 "441.jpg":4,
			 "442.jpg":4,
			 "443.jpg":4,
			 "444.jpg":6,
			 "445.jpg":6,
			 "446.jpg":6,
			 "447.jpg":6,
			 "448.jpg":6,
			 "449.jpg":6,
			 "450.jpg":6,
			 "451.jpg":6,
			 "452.jpg":6,
			 "453.jpg":6,
			 "454.jpg":6,
			 "455.jpg":6,
			 "456.jpg":6,
			 "457.jpg":6,
			 "458.jpg":6,
			 "459.jpg":6,
			 "460.jpg":6,
			 "461.jpg":6,
			 "462.jpg":5,
			 "463.jpg":5,
			 "464.jpg":5,
			 "465.jpg":5,
			 "466.jpg":5,
			 "467.jpg":5,
			 "468.jpg":5,
			 "469.jpg":5,
			 "470.jpg":5,
			 "471.jpg":5,
			 "472.jpg":5,
			 "473.jpg":7,
			 "474.jpg":7,
			 "475.jpg":7,
			 "476.jpg":7,
			 "477.jpg":7,
			 "478.jpg":7,
			 "479.jpg":7,
			 "480.jpg":7,
			 "481.jpg":7,
			 "482.jpg":7,
			 "483.jpg":7,
			 "484.jpg":7,
			 "485.jpg":7,
			 "486.jpg":7,
			 "487.jpg":7,
			 "488.jpg":7,
			 "489.jpg":7,
			 "490.jpg":7,
			 "491.jpg":7,
			 "492.jpg":7,
			 "493.jpg":7,
			 "494.jpg":7,
			 "495.jpg":7,
			 "496.jpg":7,
			 "497.jpg":7,
			 "498.jpg":7,
			 "499.jpg":7,
			 "500.jpg":7,
			 "751.jpg" :3,
		    "752.jpg" : 3,
			"753.jpg" : 3,
			"754.jpg" : 3,
			"755.jpg" : 3,
			"756.jpg" : 3,
			"757.jpg" : 3,
			"758.jpg" : 3,
			"759.jpg" : 3,
			"760.jpg" : 3,
			"761.jpg" : 3,
			"762.jpg" : 3,
			"763.jpg" : 3,
			"764.jpg" : 3,
			"765.jpg" : 3,
			"766.jpg" : 3,
			"767.jpg" : 3,
			"768.jpg" : 3,
			"769.jpg" : 3,
			"770.jpg" : 7,
			"771.jpg" : 7,
			"772.jpg" : 7,
			"773.jpg" : 7,
			"774.jpg" : 7,
			"775.jpg" : 7,
			"776.jpg" : 7,
			"777.jpg" : 7,
			"778.jpg" : 7,
			"779.jpg" : 7,
			"780.jpg" : 7,
			"781.jpg" : 7,
			"782.jpg" : 7,
			"783.jpg" : 7,
			"784.jpg" : 7,
			"785.jpg" : 7,
			"786.jpg" : 7,
			"787.jpg" : 7,
			"788.jpg" : 7,
			"789.jpg" : 7,
			"790.jpg" : 7,
			"791.jpg" : 7,
			"792.jpg" : 7,
			"793.jpg" : 7,
			"794.jpg" : 7,
			"795.jpg" : 7,
			"796.jpg" : 7,
			"797.jpg" : 7,
			"798.jpg" : 7,
			"799.jpg" : 7,
			"800.jpg" : 7,
			"801.jpg" : 7,
			"802.jpg" : 7,
			"803.jpg" : 7,
			"804.jpg" : 7,
			"805.jpg" : 7,
			"806.jpg" : 7,
			"807.jpg" : 7,
			"808.jpg" : 7,
			"809.jpg" : 7,
			"810.jpg" : 7,
			"811.jpg" : 7,
			"812.jpg" : 3,
			"813.jpg" : 3,
			"814.jpg" : 3,
			"815.jpg" : 3,
			"816.jpg" : 3,
			"817.jpg" : 3,
			"818.jpg" : 3,
			"819.jpg" : 3,
			"820.jpg" : 3,
			"821.jpg" : 3,
			"822.jpg" : 3,
			"823.jpg" : 3,
			"824.jpg" : 3,
			"825.jpg" : 3,
			"826.jpg" : 3,
			"827.jpg" : 3,
			"828.jpg" : 3,
			"829.jpg" : 3,
			"830.jpg" : 3,
			"831.jpg" : 3,
			"832.jpg" : 3,
			"833.jpg" : 3,
			"834.jpg" : 3,
			"835.jpg" : 3,
			"836.jpg" : 3,
			"837.jpg" : 3,
			"838.jpg" : 3,
			"839.jpg" : 3,
			"840.jpg" : 3,
			"841.jpg" : 3,
			"842.jpg" : 3,
			"843.jpg" : 3,
			"844.jpg" : 3,
			"845.jpg" : 3,
			"846.jpg" : 3,
			"847.jpg" : 3,
			"848.jpg" : 3,
			"849.jpg" : 3,
			"850.jpg" : 3,
			"851.jpg" : 7,
			"852.jpg" : 7,
			"853.jpg" : 7,
			"854.jpg" : 7,
			"855.jpg" : 7,
			"856.jpg" : 7,
			"857.jpg" : 7,
			"858.jpg" : 7,
			"859.jpg" : 7,
			"860.jpg" : 7,
			"861.jpg" : 7,
			"862.jpg" : 7,
			"863.jpg" : 7,
			"864.jpg" : 7,
			"865.jpg" : 7,
			"866.jpg" : 7,
			"867.jpg" : 7,
			"868.jpg" : 7,
			"869.jpg" : 7,
			"870.jpg" : 7,
			"871.jpg" : 7,
			"872.jpg" : 7,
			"873.jpg" : 7,
			"874.jpg" : 7,
			"875.jpg" : 7,
			"876.jpg" : 7,
			"877.jpg" : 7,
			"878.jpg" : 7,
			"879.jpg" : 7,
			"880.jpg" : 7,
			"881.jpg" : 7,
			"882.jpg" : 7,
			"883.jpg" : 7,
			"884.jpg" : 7,
			"885.jpg" : 7,
			"886.jpg" : 7,
			"887.jpg" : 7,
			"888.jpg" : 7,
			"889.jpg" : 7,
			"890.jpg" : 7,
			"891.jpg" : 7,
			"892.jpg" : 7,
			"893.jpg" : 7,
			"894.jpg" : 7,
			"895.jpg" : 7,
			"896.jpg" : 7,
			"897.jpg" : 7,
			"898.jpg" : 7,
			"899.jpg" : 7,
			"900.jpg" : 7,
			"901.jpg" : 7,
			"902.jpg" : 7,
			"903.jpg" : 7,
			"904.jpg" : 7,
			"905.jpg" : 7,
			"906.jpg" : 7,
			"907.jpg" : 7,
			"908.jpg" : 7,
			"909.jpg" : 7,
			"910.jpg" : 7,
			"911.jpg" : 7,
			"912.jpg" : 7,
			"913.jpg" : 7,
			"914.jpg" : 7,
			"915.jpg" : 7,
			"916.jpg" : 7,
			"917.jpg" : 7,
			"918.jpg" : 7,
			"919.jpg" : 7,
			"920.jpg" : 7,
			"921.jpg" : 7,
			"922.jpg" : 7,
			"923.jpg" : 7,
			"924.jpg" : 7,
			"925.jpg" : 7,
			"926.jpg" : 7,
			"927.jpg" : 7,
			"928.jpg" : 7,
			"929.jpg" : 4,
			"930.jpg" : 4,
			"931.jpg" : 4,
			"932.jpg" : 4,
			"933.jpg" : 4,
			"934.jpg" : 4,
			"935.jpg" : 4,
			"936.jpg" : 4,
			"937.jpg" : 4,
			"938.jpg" : 4,
			"939.jpg" : 4,
			"940.jpg" : 4,
			"941.jpg" : 4,
			"942.jpg" : 4,
			"943.jpg" : 4,
			"944.jpg" : 4,
			"945.jpg" : 4,
			"946.jpg" : 4,
			"947.jpg" : 4,
			"948.jpg" : 4,
			"949.jpg" : 4,
			"950.jpg" : 4,
			"951.jpg" : 4,
			"952.jpg" : 4,
			"953.jpg" : 4,
			"954.jpg" : 4,
			"955.jpg" : 4,
			"956.jpg" : 4,
			"957.jpg" : 4,
			"958.jpg" : 4,
			"959.jpg" : 5,
			"960.jpg" : 5,
			"961.jpg" : 5,
			"962.jpg" : 5,
			"963.jpg" : 5,
			"964.jpg" : 5,
			"965.jpg" : 5,
			"966.jpg" : 5,
			"967.jpg" : 5,
			"968.jpg" : 5,
			"969.jpg" : 5,
			"970.jpg" : 5,
			"971.jpg" : 5,
			"972.jpg" : 5,
			"973.jpg" : 5,
			"974.jpg" : 5,
			"975.jpg" : 5,
			"976.jpg" : 5,
			"977.jpg" : 5,
			"978.jpg" : 5,
			"979.jpg" : 5,
			"980.jpg" : 5,
			"981.jpg" : 5,
			"982.jpg" : 5,
			"983.jpg" : 3,
			"984.jpg" : 3,
			"985.jpg" : 3,
			"986.jpg" : 3,
			"987.jpg" : 3,
			"988.jpg" : 3,
			"989.jpg" : 3,
			"990.jpg" : 3,
			"991.jpg" : 3,
			"992.jpg" : 3,
			"993.jpg" : 3,
			"994.jpg" : 3,
			"995.jpg" : 3,
			"996.jpg" : 3,
			"997.jpg" : 3,
			"998.jpg" : 3,
			"999.jpg" : 3,
			"1000.jpg" : 3
}

# def split_dataset(v):
# 	train=[]
# 	test=[]
# 	k=len(v)
# 	k=k*1.0

# 	t_len=int(k*0.8)
# 	s_len=int(k*0.2)

# 	c=0
# 	if v is not None:
# 		for a in v:
# 			c+=1
# 			if c<=t_len:
# 				train.append(a)
# 			test.append(a)
# 	return (train,test)

new_labels={}
for k,v in label.items():
	new_labels.setdefault(v,[]).append(k)

# for k,v in new_labels.iteritems():
# 	print len(v)
# 	print


train_image=[]
test_image=[]
train_labels=[]
test_labels=[]

train_image_2=[]
test_image_2=[]
train_labels_2=[]
test_labels_2=[]
a=0

for k , v in new_labels.items():
	# (train,test)=split_dataset(v)
	train=[]
	test=[]
	k=len(v)
	k=k*1.0

	t_len=int(k*0.8)
	s_len=int(k*0.2)

	c=0
	for a in v:
		c+=1
		if c<=t_len:
			train.append(a)
		test.append(a)
	a=0
	for t in train:
		a+=1
		if a>32:
			break
		train_image.append(t)
		train_labels.append(label[t])


	a=0
	if test is not None:
		for s in test:
			a+=1
			if a>8:
				break
			test_image.append(s)
			test_labels.append(label[s])

# a=0
# for k , v in new_labels.items():
# 	# (train,test)=split_dataset(v)
# 	train=[]
# 	test=[]
# 	k=len(v)
# 	k=k*1.0

# 	t_len=int(k*0.8)
# 	s_len=int(k*0.2)

# 	c=0
# 	if v is not None:
# 		for a in v:
# 			c+=1
# 			if c<=t_len:
# 				train.append(a)
# 			test.append(a)
# 	a=0
# 	if train is not None:	
# 		for t in random.shuffle(train):
# 			a+=1
# 			if a>32:
# 				break
# 			train_image_2.append(t)
# 			train_labels_2.append(label[t])
# 	a=0
# 	if test is not None:
# 		for s in random.shuffle(test):
# 			a+=1
# 			if a>8:
# 				break
# 			test_image_2.append(s)
# 			test_labels_2.append(label[s])


# print len(train_image)
# print len(test_image)
