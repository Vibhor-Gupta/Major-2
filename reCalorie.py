import random
label =	{	
			"1.jpg" :2,
			"2.jpg" :2,
			"3.jpg" :6,
			"4.jpg" :3,
			"5.jpg" :6,
			"6.jpg" :2,
			"7.jpg" :2,
			"8.jpg" :6,
			"9.jpg" :3,
			"10.jpg" :6,
			"11.jpg" :2,
			"12.jpg" :2,
			"13.jpg" :6,
			"14.jpg" :3,
			"15.jpg" :6,
			"16.jpg" :2,
			"17.jpg" :2,
			"18.jpg" :6,
			"19.jpg" :3,
			"20.jpg" :6,
			"21.jpg" :2,
			"22.jpg" :2,
			"23.jpg" :6,
			"24.jpg" :3,
			"25.jpg" :6,
			"26.jpg" :2,
			"27.jpg" :2,
			"28.jpg" :6,
			"29.jpg" :3,
			"30.jpg" :6,
			"31.jpg" :3,
			"32.jpg" :6,
			"33.jpg" :5,
			"34.jpg" :3,
			"35.jpg" :6,
			"36.jpg" :2,
			"37.jpg" :5,
			"38.jpg" :3,
			"39.jpg" :6,
			"40.jpg" :2,
			"41.jpg" :5,
			"42.jpg" :3,
			"43.jpg" :6,
			"44.jpg" :2,
			"45.jpg" :5,
			"46.jpg" :3,
			"47.jpg" :6,
			"48.jpg" :2,
			"49.jpg" :5,
			"50.jpg" :3,
			"51.jpg" :6,
			"52.jpg" :2,
			"53.jpg" :5,
			"54.jpg" :6,
			"55.jpg" :2,
			"56.jpg" :5,
			"57.jpg" :3,
			"58.jpg" :6,
			"59.jpg" :5,
			"60.jpg" :3,
			"61.jpg" :6,
			"62.jpg" :2,
			"63.jpg" :5,
			"64.jpg" :3,
			"65.jpg" :6,
			"66.jpg" :2,
			"67.jpg" :5,
			"68.jpg" :3,
			"69.jpg" :6,
			"70.jpg" :2,
			"71.jpg" :5,
			"72.jpg" :3,
			"73.jpg" :6,
			"74.jpg" :2,
			"75.jpg" :5,
			"76.jpg" :3,
			"77.jpg" :6,
			"78.jpg" :2,
			"79.jpg" :2,
			"80.jpg" :5,
			"81.jpg" :3,
			"82.jpg" :5,
			"83.jpg" :3,
			"84.jpg" :2,
			"85.jpg" :5,
			"86.jpg" :3,
			"87.jpg" :2,
			"88.jpg" :5,
			"89.jpg" :3,
			"90.jpg" :2,
			"91.jpg" :5,
			"92.jpg" :3,
			"93.jpg" :2,
			"94.jpg" :5,
			"95.jpg" :3,
			"96.jpg" :2,
			"97.jpg" :2,
			"98.jpg" :2,
			"99.jpg" :5,
			"100.jpg" :3,
			"101.jpg" :2,
			"102.jpg" :5,
			"103.jpg" :3,
			"104.jpg" :2,
			"105.jpg" :5,
			"106.jpg" :3,
			"107.jpg" :2,
			"108.jpg" :5,
			"109.jpg" :3,
			"110.jpg" :2,
			"111.jpg" :5,
			"112.jpg" :3,
			"113.jpg" :5,
			"114.jpg" :3,
			"115.jpg" :2,
			"116.jpg" :2,
			"117.jpg" :5,
			"118.jpg" :2,
			"119.jpg" :2,
			"120.jpg" :2,
			"121.jpg" :5,
			"122.jpg" :3,
			"123.jpg" :6,
			"124.jpg" :2,
			"125.jpg" :2,
			"126.jpg" :2,
			"127.jpg" :5,
			"128.jpg" :3,
			"129.jpg" :6,
			"130.jpg" :2,
			"131.jpg" :2,
			"132.jpg" :2,
			"133.jpg" :5,
			"134.jpg" :3,
			"135.jpg" :6,
			"136.jpg" :2,
			"137.jpg" :2,
			"138.jpg" :2,
			"139.jpg" :5,
			"140.jpg" :3,
			"141.jpg" :6,
			"142.jpg" :2,
			"143.jpg" :2,
			"144.jpg" :2,
			"145.jpg" :5,
			"146.jpg" :3,
			"147.jpg" :6,
			"148.jpg" :2,
			"149.jpg" :3,
			"150.jpg" :6,
			"151.jpg" :2,
			"152.jpg" :2,
			"153.jpg" :5,
			"154.jpg" :2,
			"155.jpg" :5,
			"156.jpg" :3,
			"157.jpg" :2,
			"158.jpg" :2,
			"159.jpg" :2,
			"160.jpg" :5,
			"161.jpg" :3,
			"162.jpg" :2,
			"163.jpg" :2,
			"164.jpg" :2,
			"165.jpg" :5,
			"166.jpg" :3,
			"167.jpg" :2,
			"168.jpg" :2,
			"169.jpg" :2,
			"170.jpg" :5,
			"171.jpg" :3,
			"172.jpg" :2,
			"173.jpg" :2,
			"174.jpg" :2,
			"175.jpg" :5,
			"176.jpg" :3,
			"177.jpg" :2,
			"178.jpg" :2,
			"179.jpg" :3,
			"180.jpg" :2,
			"181.jpg" :2,
			"182.jpg" :2,
			"183.jpg" :2,
			"184.jpg" :3,
			"185.jpg" :2,
			"186.jpg" :3,
			"187.jpg" :2,
			"188.jpg" :2,
			"189.jpg" :2,
			"190.jpg" :2,
			"191.jpg" :3,
			"192.jpg" :2,
			"193.jpg" :2,
			"194.jpg" :2,
			"195.jpg" :3,
			"196.jpg" :2,
			"197.jpg" :2,
			"198.jpg" :2,
			"199.jpg" :3,
			"200.jpg" :2,
			"201.jpg" :2,
			"202.jpg" :2,
			"203.jpg" :3,
			"204.jpg" :2,
			"205.jpg" :2,
			"206.jpg" :2,
			"207.jpg" :2,
			"208.jpg" :2,
			"209.jpg" :3,
			"210.jpg" :2,
			"211.jpg" :3,
			"212.jpg" :2,
			"213.jpg" :2,
			"214.jpg" :2,
			"215.jpg" :3,
			"216.jpg" :2,
			"217.jpg" :2,
			"218.jpg" :3,
			"219.jpg" :1,
			"220.jpg" :2,
			"221.jpg" :2,
			"222.jpg" :3,
			"223.jpg" :1,
			"224.jpg" :2,
			"225.jpg" :3,
			"226.jpg" :1,
			"227.jpg" :1,
			"228.jpg" :1,
			"229.jpg" :2,
			"230.jpg" :3,
			"231.jpg" :1,
			"232.jpg" :2,
			"233.jpg" :3,
			"234.jpg" :1,
			"235.jpg" :2,
			"236.jpg" :2,
			"237.jpg" :3,
			"238.jpg" :1,
			"239.jpg" :0,
			"240.jpg" :2,
			"241.jpg" :2,
			"242.jpg" :3,
			"243.jpg" :1,
			"244.jpg" :0,
			"245.jpg" :2,
			"246.jpg" :2,
			"247.jpg" :2,
			"248.jpg" :3,
			"249.jpg" :1,
			"250.jpg" :0,
	    	 "251.jpg":2,
			 "252.jpg":2,
			 "253.jpg":2,
			 "254.jpg":3,
			 "255.jpg":1,
			 "256.jpg":2,
			 "257.jpg":0,
			 "258.jpg":2,
			 "259.jpg":2,
			 "260.jpg":2,
			 "261.jpg":3,
			 "262.jpg":0,
			 "263.jpg":2,
			 "264.jpg":3,
			 "265.jpg":2,
			 "266.jpg":2,
			 "267.jpg":2,
			 "268.jpg":3,
			 "269.jpg":2,
			 "270.jpg":2,
			 "271.jpg":3,
			 "272.jpg":2,
			 "273.jpg":2,
			 "274.jpg":2,
			 "275.jpg":2,
			 "276.jpg":3,
			 "277.jpg":0,
			 "278.jpg":2,
			 "279.jpg":2,
			 "280.jpg":3,
			 "281.jpg":0,
			 "282.jpg":2,
			 "283.jpg":2,
			 "284.jpg":0,
			 "285.jpg":2,
			 "286.jpg":0,
			 "287.jpg":2,
			 "288.jpg":2,
			 "289.jpg":0,
			 "290.jpg":2,
			 "291.jpg":2,
			 "292.jpg":3,
			 "293.jpg":0,
			 "294.jpg":2,
			 "295.jpg":3,
			 "296.jpg":2,
			 "297.jpg":2,
			 "298.jpg":2,
			 "299.jpg":2,
			 "300.jpg":3,
			 "301.jpg":2,
			 "302.jpg":2,
			 "303.jpg":2,
			 "304.jpg":3,
			 "305.jpg":0,
			 "306.jpg":2,
			 "307.jpg":2,
			 "308.jpg":2,
			 "309.jpg":3,
			 "310.jpg":0,
			 "311.jpg":2,
			 "312.jpg":2,
			 "313.jpg":3,
			 "314.jpg":0,
			 "315.jpg":2,
			 "316.jpg":2,
			 "317.jpg":0,
			 "318.jpg":2,
			 "319.jpg":2,
			 "320.jpg":0,
			 "321.jpg":2,
			 "322.jpg":0,
			 "323.jpg":2,
			 "324.jpg":2,
			 "325.jpg":2,
			 "326.jpg":2,
			 "327.jpg":2,
			 "328.jpg":2,
			 "329.jpg":2,
			 "330.jpg":2,
			 "331.jpg":2,
			 "332.jpg":6,
			 "333.jpg":6,
			 "334.jpg":6,
			 "335.jpg":6,
			 "336.jpg":4,
			 "337.jpg":6,
			 "338.jpg":4,
			 "339.jpg":6,
			 "340.jpg":4,
			 "341.jpg":4,
			 "342.jpg":4,
			 "343.jpg":4,
			 "344.jpg":4,
			 "345.jpg":4,
			 "346.jpg":4,
			 "347.jpg":4,
			 "348.jpg":4,
			 "349.jpg":4,
			 "350.jpg":2,
			 "351.jpg":2,
			 "352.jpg":4,
			 "353.jpg":2,
			 "354.jpg":4,
			 "355.jpg":2,
			 "356.jpg":4,
			 "357.jpg":2,
			 "358.jpg":4,
			 "359.jpg":2,
			 "360.jpg":4,
			 "361.jpg":4,
			 "362.jpg":2,
			 "363.jpg":4,
			 "364.jpg":2,
			 "365.jpg":4,
			 "366.jpg":2,
			 "367.jpg":4,
			 "368.jpg":2,
			 "369.jpg":4,
			 "370.jpg":2,
			 "371.jpg":4,
			 "372.jpg":4,
			 "373.jpg":2,
			 "374.jpg":2,
			 "375.jpg":2,
			 "376.jpg":2,
			 "377.jpg":2,
			 "378.jpg":2,
			 "379.jpg":3,
			 "380.jpg":3,
			 "381.jpg":3,
			 "382.jpg":3,
			 "383.jpg":3,
			 "384.jpg":3,
			 "385.jpg":3,
			 "386.jpg":3,
			 "387.jpg":3,
			 "388.jpg":3,
			 "389.jpg":3,
			 "390.jpg":3,
			 "391.jpg":3,
			 "392.jpg":3,
			 "393.jpg":3,
			 "394.jpg":3,
			 "395.jpg":3,
			 "396.jpg":3,
			 "397.jpg":3,
			 "398.jpg":3,
			 "399.jpg":3,
			 "400.jpg":3,
			 "401.jpg":3,
			 "402.jpg":3,
			 "403.jpg":3,
			 "404.jpg":3,
			 "405.jpg":3,
			 "406.jpg":3,
			 "407.jpg":3,
			 "408.jpg":3,
			 "409.jpg":3,
			 "410.jpg":3,
			 "411.jpg":3,
			 "412.jpg":3,
			 "413.jpg":3,
			 "414.jpg":3,
			 "415.jpg":3,
			 "416.jpg":3,
			 "417.jpg":3,
			 "418.jpg":3,
			 "419.jpg":3,
			 "420.jpg":3,
			 "421.jpg":3,
			 "422.jpg":3,
			 "423.jpg":3,
			 "424.jpg":3,
			 "425.jpg":3,
			 "426.jpg":3,
			 "427.jpg":3,
			 "428.jpg":3,
			 "429.jpg":3,
			 "430.jpg":3,
			 "431.jpg":3,
			 "432.jpg":3,
			 "433.jpg":3,
			 "434.jpg":3,
			 "435.jpg":3,
			 "436.jpg":3,
			 "437.jpg":3,
			 "438.jpg":3,
			 "439.jpg":3,
			 "440.jpg":3,
			 "441.jpg":3,
			 "442.jpg":3,
			 "443.jpg":3,
			 "444.jpg":5,
			 "445.jpg":5,
			 "446.jpg":5,
			 "447.jpg":5,
			 "448.jpg":5,
			 "449.jpg":5,
			 "450.jpg":5,
			 "451.jpg":5,
			 "452.jpg":5,
			 "453.jpg":5,
			 "454.jpg":5,
			 "455.jpg":5,
			 "456.jpg":5,
			 "457.jpg":5,
			 "458.jpg":5,
			 "459.jpg":5,
			 "460.jpg":5,
			 "461.jpg":5,
			 "462.jpg":4,
			 "463.jpg":4,
			 "464.jpg":4,
			 "465.jpg":4,
			 "466.jpg":4,
			 "467.jpg":4,
			 "468.jpg":4,
			 "469.jpg":4,
			 "470.jpg":4,
			 "471.jpg":4,
			 "472.jpg":4,
			 "473.jpg":6,
			 "474.jpg":6,
			 "475.jpg":6,
			 "476.jpg":6,
			 "477.jpg":6,
			 "478.jpg":6,
			 "479.jpg":6,
			 "480.jpg":6,
			 "481.jpg":6,
			 "482.jpg":6,
			 "483.jpg":6,
			 "484.jpg":6,
			 "485.jpg":6,
			 "486.jpg":6,
			 "487.jpg":6,
			 "488.jpg":6,
			 "489.jpg":6,
			 "490.jpg":6,
			 "491.jpg":6,
			 "492.jpg":6,
			 "493.jpg":6,
			 "494.jpg":6,
			 "495.jpg":6,
			 "496.jpg":6,
			 "497.jpg":6,
			 "498.jpg":6,
			 "499.jpg":6,
			 "500.jpg":6,
			 "501.jpg":6,
			 "502.jpg":6,
			 "503.jpg":5,
			 "504.jpg":5,
			 "505.jpg":5,
			 "506.jpg":5,
			 "507.jpg":5,
			 "508.jpg":5,
			 "509.jpg":5,
			 "510.jpg":5,
			 "511.jpg":5,
			 "512.jpg":5,
			 "513.jpg":5,
			 "514.jpg":5,
			 "515.jpg":5,
			 "516.jpg":5,
			 "517.jpg":5,
			 "518.jpg":5,
			 "519.jpg":5,
			 "520.jpg":5,
			 "521.jpg":6,
			 "522.jpg":6,
			 "523.jpg":6,
			 "524.jpg":6,
			 "525.jpg":6,
			 "526.jpg":6,
			 "527.jpg":6,
			 "528.jpg":6,
			 "529.jpg":6,
			 "530.jpg":6,
			 "531.jpg":6,
			 "532.jpg":6,
			 "533.jpg":6,
			 "534.jpg":6,
			 "535.jpg":6,
			 "536.jpg":6,
			 "537.jpg":6,
			 "538.jpg":6,
			 "539.jpg":6,
			 "540.jpg":6,
			 "541.jpg":6,
			 "542.jpg":6,
			 "543.jpg":6,
			 "544.jpg":6,
			 "545.jpg":6,
			 "546.jpg":6,
			 "547.jpg":6,
			 "548.jpg":6,
			 "549.jpg":6,
			 "550.jpg":6,
			 "551.jpg":5,
			 "552.jpg":5,
			 "553.jpg":5,
			 "554.jpg":5,
			 "555.jpg":5,
			 "556.jpg":5,
			 "557.jpg":5,
			 "558.jpg":5,
			 "559.jpg":5,
			 "560.jpg":5,
			 "561.jpg":5,
			 "562.jpg":5,
			 "563.jpg":5,
			 "564.jpg":5,
			 "565.jpg":5,
			 "566.jpg":5,
			 "567.jpg":5,
			 "568.jpg":5,
			 "569.jpg":6,
			 "570.jpg":6,
			 "571.jpg":6,
			 "572.jpg":6,
			 "573.jpg":6,
			 "574.jpg":6,
			 "575.jpg":6,
			 "576.jpg":6,
			 "577.jpg":6,
			 "578.jpg":6,
			 "579.jpg":6,
			 "580.jpg":6,
			 "581.jpg":6,
			 "582.jpg":6,
			 "583.jpg":6,
			 "584.jpg":6,
			 "585.jpg":6,
			 "586.jpg":0,
			 "587.jpg":0,
			 "588.jpg":0,
			 "589.jpg":0,
			 "590.jpg":0,
			 "591.jpg":0,
			 "592.jpg":0,
			 "593.jpg":0,
			 "594.jpg":0,
			 "595.jpg":0,
			 "596.jpg":0,
			 "597.jpg":0,
			 "598.jpg":1,
			 "599.jpg":1,
			 "600.jpg":1,
			 "601.jpg":1,
			 "602.jpg":1,
			 "603.jpg":1,
			 "604.jpg":5,
			 "605.jpg":5,
			 "606.jpg":5,
			 "607.jpg":5,
			 "608.jpg":5,
			 "609.jpg":5,
			 "610.jpg":5,
			 "611.jpg":5,
			 "612.jpg":5,
			 "613.jpg":5,
			 "614.jpg":5,
			 "615.jpg":5,
			 "616.jpg":6,
			 "617.jpg":6,
			 "618.jpg":6,
			 "619.jpg":6,
			 "620.jpg":6,
			 "621.jpg":6,
			 "622.jpg":5,
			 "623.jpg":5,
			 "624.jpg":5,
			 "625.jpg":5,
			 "626.jpg":5,
			 "627.jpg":5,
			 "628.jpg":6,
			 "629.jpg":6,
			 "630.jpg":6,
			 "631.jpg":6,
			 "632.jpg":6,
			 "633.jpg":6,
			 "634.jpg":6,
			 "635.jpg":6,
			 "636.jpg":6,
			 "637.jpg":6,
			 "638.jpg":6,
			 "639.jpg":6,
			 "640.jpg":6,
			 "641.jpg":6,
			 "642.jpg":6,
			 "643.jpg":6,
			 "644.jpg":6,
			 "645.jpg":6,
			 "646.jpg":6,
			 "647.jpg":6,
			 "648.jpg":6,
			 "649.jpg":6,
			 "650.jpg":6,
			 "651.jpg":6,
			 "652.jpg":6,
			 "653.jpg":6,
			 "654.jpg":6,
			 "655.jpg":6,
			 "656.jpg":5,
			 "657.jpg":5,
			 "658.jpg":5,
			 "659.jpg":5,
			 "660.jpg":5,
			 "661.jpg":5,
			 "662.jpg":5,
			 "663.jpg":5,
			 "664.jpg":5,
			 "665.jpg":5,
			 "666.jpg":5,
			 "667.jpg":5,
			 "668.jpg":6,
			 "669.jpg":6,
			 "670.jpg":6,
			 "671.jpg":6,
			 "672.jpg":6,
			 "673.jpg":6,
			 "674.jpg":0,
			 "675.jpg":0,
			 "676.jpg":0,
			 "677.jpg":0,
			 "678.jpg":0,
			 "679.jpg":0,
			 "680.jpg":6,
			 "681.jpg":6,
			 "682.jpg":6,
			 "683.jpg":6,
			 "684.jpg":6,
			 "685.jpg":6,
			 "686.jpg":1,
			 "687.jpg":1,
			 "688.jpg":1,
			 "689.jpg":1,
			 "690.jpg":1,
			 "691.jpg":1,
			 "692.jpg":0,
			 "693.jpg":0,
			 "694.jpg":0,
			 "695.jpg":0,
			 "696.jpg":0,
			 "697.jpg":0,
			 "698.jpg":2,
			 "699.jpg":2,
			 "700.jpg":2,
			 "701.jpg":2,
			 "702.jpg":2,
			 "703.jpg":2,
			 "704.jpg":2,
			 "705.jpg":2,
			 "706.jpg":2,
			 "707.jpg":2,
			 "708.jpg":2,
			 "709.jpg":2,
			 "710.jpg":2,
			 "711.jpg":2,
			 "712.jpg":2,
			 "713.jpg":2,
			 "714.jpg":2,
			 "715.jpg":2,
			 "716.jpg":2,
			 "717.jpg":2,
			 "718.jpg":2,
			 "719.jpg":2,
			 "720.jpg":2,
			 "721.jpg":2,
			 "722.jpg":2,
			 "723.jpg":2,
			 "724.jpg":2,
			 "725.jpg":2,
			 "726.jpg":2,
			 "727.jpg":2,
			 "728.jpg":2,
			 "729.jpg":2,
			 "730.jpg":2,
			 "731.jpg":2,
			 "732.jpg":2,
			 "733.jpg":2,
			 "734.jpg":2,
			 "735.jpg":2,
			 "736.jpg":2,
			 "737.jpg":2,
			 "738.jpg":2,
			 "739.jpg":2,
			 "740.jpg":2,
			 "741.jpg":2,
			 "742.jpg":2,
			 "743.jpg":2,
			 "744.jpg":2,
			 "745.jpg":2,
			 "746.jpg":2,
			 "747.jpg":2,
			 "748.jpg":2,
			 "749.jpg":2,
			 "750.jpg":2,
			 "751.jpg" :2,
		    "752.jpg" :2,
			"753.jpg" :2,
			"754.jpg" :2,
			"755.jpg" :2,
			"756.jpg" :2,
			"757.jpg" :2,
			"758.jpg" :2,
			"759.jpg" :2,
			"760.jpg" :2,
			"761.jpg" :2,
			"762.jpg" :2,
			"763.jpg" :2,
			"764.jpg" :2,
			"765.jpg" :2,
			"766.jpg" :2,
			"767.jpg" :2,
			"768.jpg" :2,
			"769.jpg" :2,
			"770.jpg" :6,
			"771.jpg" :6,
			"772.jpg" :6,
			"773.jpg" :6,
			"774.jpg" :6,
			"775.jpg" :6,
			"776.jpg" :6,
			"777.jpg" :6,
			"778.jpg" :6,
			"779.jpg" :6,
			"780.jpg" :6,
			"781.jpg" :6,
			"782.jpg" :6,
			"783.jpg" :6,
			"784.jpg" :6,
			"785.jpg" :6,
			"786.jpg" :6,
			"787.jpg" :6,
			"788.jpg" :6,
			"789.jpg" :6,
			"790.jpg" :6,
			"791.jpg" :6,
			"792.jpg" :6,
			"793.jpg" :6,
			"794.jpg" :6,
			"795.jpg" :6,
			"796.jpg" :6,
			"797.jpg" :6,
			"798.jpg" :6,
			"799.jpg" :6,
			"800.jpg" :6,
			"801.jpg" :6,
			"802.jpg" :6,
			"803.jpg" :6,
			"804.jpg" :6,
			"805.jpg" :6,
			"806.jpg" :6,
			"807.jpg" :6,
			"808.jpg" :6,
			"809.jpg" :6,
			"810.jpg" :6,
			"811.jpg" :6,
			"812.jpg" :2,
			"813.jpg" :2,
			"814.jpg" :2,
			"815.jpg" :2,
			"816.jpg" :2,
			"817.jpg" :2,
			"818.jpg" :2,
			"819.jpg" :2,
			"820.jpg" :2,
			"821.jpg" :2,
			"822.jpg" :2,
			"823.jpg" :2,
			"824.jpg" :2,
			"825.jpg" :2,
			"826.jpg" :2,
			"827.jpg" :2,
			"828.jpg" :2,
			"829.jpg" :2,
			"830.jpg" :2,
			"831.jpg" :2,
			"832.jpg" :2,
			"833.jpg" :2,
			"834.jpg" :2,
			"835.jpg" :2,
			"836.jpg" :2,
			"837.jpg" :2,
			"838.jpg" :2,
			"839.jpg" :2,
			"840.jpg" :2,
			"841.jpg" :2,
			"842.jpg" :2,
			"843.jpg" :2,
			"844.jpg" :2,
			"845.jpg" :2,
			"846.jpg" :2,
			"847.jpg" :2,
			"848.jpg" :2,
			"849.jpg" :2,
			"850.jpg" :2,
			"851.jpg" :6,
			"852.jpg" :6,
			"853.jpg" :6,
			"854.jpg" :6,
			"855.jpg" :6,
			"856.jpg" :6,
			"857.jpg" :6,
			"858.jpg" :6,
			"859.jpg" :6,
			"860.jpg" :6,
			"861.jpg" :6,
			"862.jpg" :6,
			"863.jpg" :6,
			"864.jpg" :6,
			"865.jpg" :6,
			"866.jpg" :6,
			"867.jpg" :6,
			"868.jpg" :6,
			"869.jpg" :6,
			"870.jpg" :6,
			"871.jpg" :6,
			"872.jpg" :6,
			"873.jpg" :6,
			"874.jpg" :6,
			"875.jpg" :6,
			"876.jpg" :6,
			"877.jpg" :6,
			"878.jpg" :6,
			"879.jpg" :6,
			"880.jpg" :6,
			"881.jpg" :6,
			"882.jpg" :6,
			"883.jpg" :6,
			"884.jpg" :6,
			"885.jpg" :6,
			"886.jpg" :6,
			"887.jpg" :6,
			"888.jpg" :6,
			"889.jpg" :6,
			"890.jpg" :6,
			"891.jpg" :6,
			"892.jpg" :6,
			"893.jpg" :6,
			"894.jpg" :6,
			"895.jpg" :6,
			"896.jpg" :6,
			"897.jpg" :6,
			"898.jpg" :6,
			"899.jpg" :6,
			"900.jpg" :6,
			"901.jpg" :6,
			"902.jpg" :6,
			"903.jpg" :6,
			"904.jpg" :6,
			"905.jpg" :6,
			"906.jpg" :6,
			"907.jpg" :6,
			"908.jpg" :6,
			"909.jpg" :6,
			"910.jpg" :6,
			"911.jpg" :6,
			"912.jpg" :6,
			"913.jpg" :6,
			"914.jpg" :6,
			"915.jpg" :6,
			"916.jpg" :6,
			"917.jpg" :6,
			"918.jpg" :6,
			"919.jpg" :6,
			"920.jpg" :6,
			"921.jpg" :6,
			"922.jpg" :6,
			"923.jpg" :6,
			"924.jpg" :6,
			"925.jpg" :6,
			"926.jpg" :6,
			"927.jpg" :6,
			"928.jpg" :6,
			"929.jpg" :3,
			"930.jpg" :3,
			"931.jpg" :3,
			"932.jpg" :3,
			"933.jpg" :3,
			"934.jpg" :3,
			"935.jpg" :3,
			"936.jpg" :3,
			"937.jpg" :3,
			"938.jpg" :3,
			"939.jpg" :3,
			"940.jpg" :3,
			"941.jpg" :3,
			"942.jpg" :3,
			"943.jpg" :3,
			"944.jpg" :3,
			"945.jpg" :3,
			"946.jpg" :3,
			"947.jpg" :3,
			"948.jpg" :3,
			"949.jpg" :3,
			"950.jpg" :3,
			"951.jpg" :3,
			"952.jpg" :3,
			"953.jpg" :3,
			"954.jpg" :3,
			"955.jpg" :3,
			"956.jpg" :3,
			"957.jpg" :3,
			"958.jpg" :3,
			"959.jpg" :4,
			"960.jpg" :4,
			"961.jpg" :4,
			"962.jpg" :4,
			"963.jpg" :4,
			"964.jpg" :4,
			"965.jpg" :4,
			"966.jpg" :4,
			"967.jpg" :4,
			"968.jpg" :4,
			"969.jpg" :4,
			"970.jpg" :4,
			"971.jpg" :4,
			"972.jpg" :4,
			"973.jpg" :4,
			"974.jpg" :4,
			"975.jpg" :4,
			"976.jpg" :4,
			"977.jpg" :4,
			"978.jpg" :4,
			"979.jpg" :4,
			"980.jpg" :4,
			"981.jpg" :4,
			"982.jpg" :4,
			"983.jpg" :2,
			"984.jpg" :2,
			"985.jpg" :2,
			"986.jpg" :2,
			"987.jpg" :2,
			"988.jpg" :2,
			"989.jpg" :2,
			"990.jpg" :2,
			"991.jpg" :2,
			"992.jpg" :2,
			"993.jpg" :2,
			"994.jpg" :2,
			"995.jpg" :2,
			"996.jpg" :2,
			"997.jpg" :2,
			"998.jpg" :2,
			"999.jpg" :2,
			"1000.jpg" :2
}


def calarieCalculation():
	cal = {}
	val=random.randint(1,5)
	for k,v in label.iteritems():
		if v==0:
			cal[k]=600+val
		elif v==1:
			cal[k]=560+val
		elif v==2:
			cal[k]=200+val
		elif v==3:
			cal[k]=300+val
		elif v==4:
			cal[k]=390+val
		elif v==5:
			cal[k]=140+val
		else:
			cal[k]=530+val
	return cal