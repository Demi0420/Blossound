# config.py

COLOR_TONE_DICT = {
    "#FFFFFF": "C-b",  # 1
    "#000000": "C-s",  # 2
    "#FFFDD0": "#C-b", # 3
    "#8B0000": "#C-s", # 4
    "#008000": "D-b",  # 5
    "#013220": "d-s",  # 6
    "#FFFFE0": "bE-b", # 7
    "#7FFF00": "be-s", # 8
    "#FFFF00": "E-b",  # 9
    "#00008B": "e-s",  # 10
    "#FFDAB9": "F-b",  # 11
    "#4B0082": "f-s",  # 12
    "#FF0000": "#F-b", # 13
    "#191970": "#f-s", # 14
    "#FFA500": "G-b",  # 15
    "#1E90FF": "g-s",  # 16
    "#B8860B": "bA-b", # 17
    "#ADD8E6": "#g-s", # 18
    "#FFD700": "A-b",  # 19
    "#0000FF": "a-s",  # 20
    "#FFB6C1": "bB-b", # 21
    "#654321": "bb-s", # 22
    "#C0C0C0": "B-b",  # 23
    "#696969": "b-s"   # 24
}

COLOR_TO_TONE_MAP = {
  "C-b": {
    "description": "#FFFFFF", # 纯白色
    "tempo": 85,
    "key": "C",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["C", "D", "E", "F", "G", "A", "B", "C"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["C", "F", "G", "C"]
  },
  "C-s": {
    "description": "#000000", # 纯黑色
    "tempo": 85,
    "key": "C",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["C", "D", "E-", "F", "G", "A-", "B-", "C"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["Cm", "Fm", "G", "Cm"]
  },
  "#C-b": {
    "description": "#FFFDD0", # 奶白色
    "tempo": 85,
    "key": "C#",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["C#", "D#", "E#", "F#", "G#", "A#", "B#", "C#"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["C#", "F#", "G#", "C#"]
  },
  "#C-s": {
    "description": "#8B0000", # 深红色
    "tempo": 85,
    "key": "C#",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["C#", "D#", "E", "F#", "G#", "A", "B", "C#"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["C#m", "F#m", "G#", "C#m"]
  },
  "D-b": {
    "description": "#008000", # 纯绿色
    "tempo": 85,
    "key": "D",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["D", "E", "F#", "G", "A", "B", "C#", "D"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["D", "G", "A", "D"]
  },
  "d-s": {
    "description": "#013220", # 墨绿色
    "tempo": 85,
    "key": "D",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["D", "E", "F", "G", "A", "B-", "C", "D"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["Dm", "Gm", "A", "Dm"]
  },
  "bE-b": {
    "description": "#FFFFE0", # 淡黄色
    "tempo": 85,
    "key": "Eb",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["E-", "F", "G", "A-", "B-", "C", "D", "E-"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["E-", "A-", "B-", "E-"]
  },
  "be-s": {
    "description": "#7FFF00", # 黄绿色
    "tempo": 85,
    "key": "Eb",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["E-", "F", "G-", "A-", "B-", "C-", "D-", "E-"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["E-m", "A-m", "B-", "E-m"]
  },
  "E-b": {
    "description": "#FFFF00", # 纯黄色
    "tempo": 85,
    "key": "E",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["E", "F#", "G#", "A", "B", "C#", "D#", "E"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["E", "A", "B", "E"],
  },
  "e-s": {
    "description": "#00008B", # 深蓝色
    "tempo": 85,
    "key": "E",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["E", "F#", "G", "A", "B", "C", "D", "E"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["Em", "Am", "B", "Em"]
  },
  "F-b": {
    "description": "#FFDAB9", # 淡橙色
    "tempo": 85,
    "key": "F",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["F", "G", "A", "B-", "C", "D", "E", "F"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["F", "B-", "C", "F"]
  },
  "f-s": {
    "description": "#4B0082", # 深紫色
    "tempo": 85,
    "key": "F",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["F", "G", "A-", "B-", "C", "D-", "E-", "F"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["Fm", "B-m", "C", "Fm"]
  },
  "#F-b": {
    "description": "#FF0000", # 鲜红色
    "tempo": 85,
    "key": "F#",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["F#", "G#", "A#", "B", "C#", "D#", "E#", "F#"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["F#", "B", "C#", "F#"]
  },
  "#f-s": {
    "description": "#191970", # 午夜蓝
    "tempo": 85,
    "key": "F#",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["F#", "G#", "A", "B", "C#", "D", "E", "F#"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["F#m", "Bm", "C#", "F#m"]
  },
  "G-b": {
    "description": "#FFA500", # 橘色
    "tempo": 85,
    "key": "G",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["G", "A", "B", "C", "D", "E", "F#", "G"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["G", "C", "D", "G"],
  },
  "g-s": {
    "description": "#1E90FF", # 海蓝色
    "tempo": 85,
    "key": "G",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["G", "A", "B-", "C", "D", "E-", "F", "G"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["Gm", "Cm", "D", "Gm"]
  },
  "bA-b": {
    "description": "#B8860B", # 深金色
    "tempo": 85,
    "key": "Ab",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["A-", "B-", "C", "D-", "E-", "F", "G", "A-"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["A-", "D-", "E-", "A-"]
  },
  "#g-s": {
    "description": "#ADD8E6", # 淡蓝色
    "tempo": 85,
    "key": "G#",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["G#", "A#", "B", "C#", "D#", "E#", "F#", "G#"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["G#m", "C#m", "D#", "G#m"]
  },
  "A-b": {
    "description": "#FFD700", # 金黄色
    "tempo": 85,
    "key": "A",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["A", "B", "C#", "D", "E", "F#", "G#", "A"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["A", "D", "E", "A"]
  },
  "a-s": {
    "description": "#0000FF", # 纯蓝色
    "tempo": 85,
    "key": "A",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["A", "B", "C", "D", "E", "F", "G", "A"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["Am", "Dm", "E", "Am"]
  },
  "bB-b": {
    "description": "#FFB6C1", # 淡粉色
    "tempo": 85,
    "key": "Bb",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["B-", "C", "D", "E-", "F", "G", "A", "B-"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["B-", "E-", "F", "B-"]
  },
  "bb-s": {
    "description": "#654321", # 深棕色
    "tempo": 85,
    "key": "Bb",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["B-", "C", "D-", "E-", "F", "G-", "A-", "B-"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["B-m", "E-m", "F", "B-m"]
  },
  "B-b": {
    "description": "#C0C0C0", # 银白色
    "tempo": 85,
    "key": "B",
    "scale_mode": "major",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["B", "C#", "D#", "E", "F#", "G#", "A#", "B"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["B", "E", "F#", "B"]
  },
  "b-s": {
    "description": "#696969",  # 暗灰色
    "tempo": 85,
    "key": "B",
    "scale_mode": "minor",
    "base_instrument": "Acoustic Grand Piano",
    "scale": ["B", "C#", "D", "E", "F#", "G", "A", "B"],
    "rhythm_pattern": [1, 1, 1, 1],
    "chord_progression": ["Bm", "Em", "F#", "Bm"]
  }
}

COLOR_MAP = {
    "V":   ["#D40045","#EE0026","#FD1A1C","#FE4118","#FF590B","#FF7F00","#FFCC00","#FFE600","#CCE700","#99CF15","#66B82B","#33A23D","#008F62","#008678","#007A87","#055D87","#093F86","#0F218B","#1D1A88","#281285","#340C81","#56007D","#770071","#AF0065"], # vidid tone
	"b":   ["#ED3B6B","#FA344D","#FC3633","#FC4E33","#FF6E2B","#FF9913","#FFCB1F","#FFF231","#CDE52F","#99D02C","#55A73B","#32A65D","#2DA380","#1AA28E","#1FB3B3","#1C86AE","#2B78B0","#396BB0","#5468AD","#6A64AE","#8561AB","#A459AB","#C75BB1","#DF4C93"], # bright tone
	"s":   ["#B01040","#CA1028","#CC2211","#CC4613","#D45F10","#D97610","#D19711","#CCB914","#B3B514","#8CA114","#41941E","#28853F","#287A52","#297364","#26707B","#205B85","#224A87","#243B8B","#241F86","#3D1C84","#4E2283","#5F2883","#8C1D84","#9A0F50"], # strong tone
	"dp":  ["#870042","#9D002B","#A20715","#A51200","#A42F03","#A24A02","#A46603","#A48204","#949110","#518517","#307A25","#306F42","#186A53","#025865","#034F69","#04436E","#05426F","#073E74","#152A6B","#232266","#3F1B63","#531560","#690C5C","#75004F"], # deep tone
	"lt":  ["#EE7296","#FB7482","#FA7272","#FB8071","#FA996F","#FDB56D","#FCD474","#FEF27A","#DDED71","#B3DE6A","#9AD47F","#7FC97E","#72C591","#66C1AF","#66C4C4","#67B1CA","#67A9C9","#689ECA","#7288C2","#817DBA","#9678B8","#B173B6","#C972B6","#E170A4"], # light tone
	"sf":  ["#BD576F","#C95F6B","#CF5E5A","#D77957","#D6763A","#D89048","#D29F34","#CCBA4C","#C0B647","#B3B140","#79B055","#66AC78","#5BA37E","#4E9B87","#4E9995","#4F8B96","#4E7592","#516691","#535A90","#5C5791","#77568F","#8B5587","#9E5485","#B05076"], # soft tone
	"d":   ["#8C355F","#994052","#A6424C","#B24443","#B34D3E","#B25939","#A66E3D","#997F42","#8C8946","#757E47","#678049","#5A814C","#39764D","#2A6A69","#256B75","#1D6283","#204F79","#214275","#2E3A76","#39367B","#493278","#5F3179","#772D7A","#802A69"], # dull tone
	"dk":  ["#632534","#632A31","#6B2B29","#743526","#6E3D1F","#6B4919","#695018","#6A5B18","#6E6E26","#56561A","#506B3E","#355935","#28523A","#1E4B44","#154D4E","#0E4250","#123B4F","#163450","#222A4E","#312C4C","#3E2E49","#4A304B","#57304B","#643142"], # dark tone
	"p":   ["#EEAFCE","#FBB4C4","#FAB6B5","#FDCDB7","#FBD8B0","#FEE6AA","#FCF1AF","#FEFFB3","#EEFAB2","#E6F5B0","#D9F6C0","#CCEAC4","#C0EBCD","#B3E2D8","#B4DDDF","#B4D7DD","#B5D2E0","#B3CEE3","#B4C2DD","#B2B6D9","#BCB2D5","#CAB2D6","#DAAFDC","#E4ADD5"], # pale tone
	"ltg": ["#C99FB3","#D7A4B5","#D6A9A4","#D7AFA7","#D9B59F","#D8BA96","#D9C098","#D9C69B","#C5CB9B","#AAC09A","#A0BD9E","#9EBCA4","#99BAA7","#92B8AD","#91B8B7","#91AFBA","#92A9B9","#91A4B5","#9199B0","#9191AD","#9C93AE","#A997B1","#B89AB6","#C09FB4"], # light grayish tone
	"g":   ["#6B455A","#7D4F5A","#7C575E","#7D5F61","#7E6261","#7C6764","#7C6A5E","#7E6F5A","#72755A","#636F5B","#586E57","#476C5B","#416863","#395B64","#38555D","#384E5C","#38475A","#394158","#353654","#3F3051","#463353","#4A3753","#553857","#5B3A55"], # grayish tone
	"dkg": ["#3C2D30","#3A2B2E","#3B2B2C","#3A2C2B","#40322F","#463B35","#453B31","#47402C","#42412F","#3E3F31","#2C382A","#24332C","#23342E","#253532","#253535","#283639","#232C33","#212832","#242331","#282530","#2A2730","#2D2A31","#362C34","#392D31"]  # dark grayish tone
}

TONE_TO_SCALE = {
    "V": {
        "tempo": 140,           
        "key": "C",             
        "scale_mode": "major",
        "instrument": "Acoustic Grand Piano",
        "scale": ["B-", "C", "D-", "E-", "F", "G-", "A-", "B-"]
    },
    "b": {
        "tempo": 130,
        "key": "G",
        "scale_mode": "major",
        "instrument": "Acoustic Grand Piano",
        "scale": ["C", "D", "E-", "F", "G", "A-", "B-", "C"]
    },
    "s": {
        "tempo": 120,
        "key": "A",
        "scale_mode": "minor",
        "instrument": "Acoustic Grand Piano",
        "scale": ["D", "E-", "F", "G", "A-", "B-", "C", "D"]
    },
    "dp": {
        "tempo": 100,
        "key": "D",
        "scale_mode": "minor",
        "instrument": "Acoustic Grand Piano",
        "scale": ["C", "D-", "E-", "F", "G-", "A-", "B-", "C"]
    },
    "lt": {
        "tempo": 90,
        "key": "F",
        "scale_mode": "major",
        "instrument": "Acoustic Grand Piano",
        "scale": ["F", "G", "A-", "B-", "C", "D-", "E-", "F"]
    },
    "sf": {
        "tempo": 80,
        "key": "E",
        "scale_mode": "major",
        "instrument": "Acoustic Grand Piano",
        "scale": ["A", "B-", "C", "D", "E", "F", "G", "A"]
    },
    "d": {
        "tempo": 70,
        "key": "G",
        "scale_mode": "minor",
        "instrument": "Acoustic Grand Piano",
        "scale": ["G", "A", "B-", "C", "D", "E-", "F", "G"]
    },
    "dk": {
        "tempo": 60,
        "key": "C",
        "scale_mode": "minor",
        "instrument": "Acoustic Grand Piano",
        "scale": ["B", "C", "D", "E", "F", "G", "A", "B"]
    },
    "p": {
        "tempo": 85,
        "key": "A",
        "scale_mode": "major",
        "instrument": "Acoustic Grand Piano",
        "scale": ["C", "D", "E", "F", "G", "A", "B", "C"]
    },
    "ltg": {
        "tempo": 100,
        "key": "C",
        "scale_mode": "major",
        "instrument": "Acoustic Grand Piano",
        "scale": ["G", "A", "B-", "C", "D", "E-", "F", "G"]
    },
    "g": {
        "tempo": 60,
        "key": "D",
        "scale_mode": "major",
        "instrument": "Acoustic Grand Piano",
        "scale": ["D", "E", "F", "G", "A", "B-", "C", "D"]
    },
    "dkg": {
        "tempo": 55,
        "key": "E",
        "scale_mode": "minor",
        "instrument": "Acoustic Grand Piano",
        "scale": ["E", "F", "G", "A", "B-", "C", "D", "E"]
    }
}

BASIC_OCTAVE = 4
LEFT_OCTAVE = 3