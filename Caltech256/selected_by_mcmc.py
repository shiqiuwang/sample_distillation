# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/17 16:37
@Author  : Qiushi Wang
@FileName: selected_by_mcmc.py
@Software: PyCharm
'''

import numpy as np
from sklearn.cluster import MeanShift
from collections import Counter
from mcmc import MC

T = 43000  # 模拟次数

label_0_loss_to_img = {}
label_1_loss_to_img = {}
label_2_loss_to_img = {}
label_3_loss_to_img = {}
label_4_loss_to_img = {}
label_5_loss_to_img = {}
label_6_loss_to_img = {}
label_7_loss_to_img = {}
label_8_loss_to_img = {}
label_9_loss_to_img = {}
label_10_loss_to_img = {}
label_11_loss_to_img = {}
label_12_loss_to_img = {}
label_13_loss_to_img = {}
label_14_loss_to_img = {}
label_15_loss_to_img = {}
label_16_loss_to_img = {}
label_17_loss_to_img = {}
label_18_loss_to_img = {}
label_19_loss_to_img = {}
label_20_loss_to_img = {}
label_21_loss_to_img = {}
label_22_loss_to_img = {}
label_23_loss_to_img = {}
label_24_loss_to_img = {}
label_25_loss_to_img = {}
label_26_loss_to_img = {}
label_27_loss_to_img = {}
label_28_loss_to_img = {}
label_29_loss_to_img = {}
label_30_loss_to_img = {}
label_31_loss_to_img = {}
label_32_loss_to_img = {}
label_33_loss_to_img = {}
label_34_loss_to_img = {}
label_35_loss_to_img = {}
label_36_loss_to_img = {}
label_37_loss_to_img = {}
label_38_loss_to_img = {}
label_39_loss_to_img = {}
label_40_loss_to_img = {}
label_41_loss_to_img = {}
label_42_loss_to_img = {}
label_43_loss_to_img = {}
label_44_loss_to_img = {}
label_45_loss_to_img = {}
label_46_loss_to_img = {}
label_47_loss_to_img = {}
label_48_loss_to_img = {}
label_49_loss_to_img = {}
label_50_loss_to_img = {}
label_51_loss_to_img = {}
label_52_loss_to_img = {}
label_53_loss_to_img = {}
label_54_loss_to_img = {}
label_55_loss_to_img = {}
label_56_loss_to_img = {}
label_57_loss_to_img = {}
label_58_loss_to_img = {}
label_59_loss_to_img = {}
label_60_loss_to_img = {}
label_61_loss_to_img = {}
label_62_loss_to_img = {}
label_63_loss_to_img = {}
label_64_loss_to_img = {}
label_65_loss_to_img = {}
label_66_loss_to_img = {}
label_67_loss_to_img = {}
label_68_loss_to_img = {}
label_69_loss_to_img = {}
label_70_loss_to_img = {}
label_71_loss_to_img = {}
label_72_loss_to_img = {}
label_73_loss_to_img = {}
label_74_loss_to_img = {}
label_75_loss_to_img = {}
label_76_loss_to_img = {}
label_77_loss_to_img = {}
label_78_loss_to_img = {}
label_79_loss_to_img = {}
label_80_loss_to_img = {}
label_81_loss_to_img = {}
label_82_loss_to_img = {}
label_83_loss_to_img = {}
label_84_loss_to_img = {}
label_85_loss_to_img = {}
label_86_loss_to_img = {}
label_87_loss_to_img = {}
label_88_loss_to_img = {}
label_89_loss_to_img = {}
label_90_loss_to_img = {}
label_91_loss_to_img = {}
label_92_loss_to_img = {}
label_93_loss_to_img = {}
label_94_loss_to_img = {}
label_95_loss_to_img = {}
label_96_loss_to_img = {}
label_97_loss_to_img = {}
label_98_loss_to_img = {}
label_99_loss_to_img = {}
label_100_loss_to_img = {}
label_101_loss_to_img = {}
label_102_loss_to_img = {}
label_103_loss_to_img = {}
label_104_loss_to_img = {}
label_105_loss_to_img = {}
label_106_loss_to_img = {}
label_107_loss_to_img = {}
label_108_loss_to_img = {}
label_109_loss_to_img = {}
label_110_loss_to_img = {}
label_111_loss_to_img = {}
label_112_loss_to_img = {}
label_113_loss_to_img = {}
label_114_loss_to_img = {}
label_115_loss_to_img = {}
label_116_loss_to_img = {}
label_117_loss_to_img = {}
label_118_loss_to_img = {}
label_119_loss_to_img = {}
label_120_loss_to_img = {}
label_121_loss_to_img = {}
label_122_loss_to_img = {}
label_123_loss_to_img = {}
label_124_loss_to_img = {}
label_125_loss_to_img = {}
label_126_loss_to_img = {}
label_127_loss_to_img = {}
label_128_loss_to_img = {}

label_129_loss_to_img = {}
label_130_loss_to_img = {}
label_131_loss_to_img = {}
label_132_loss_to_img = {}
label_133_loss_to_img = {}
label_134_loss_to_img = {}
label_135_loss_to_img = {}
label_136_loss_to_img = {}
label_137_loss_to_img = {}
label_138_loss_to_img = {}
label_139_loss_to_img = {}
label_140_loss_to_img = {}
label_141_loss_to_img = {}
label_142_loss_to_img = {}
label_143_loss_to_img = {}
label_144_loss_to_img = {}
label_145_loss_to_img = {}
label_146_loss_to_img = {}
label_147_loss_to_img = {}
label_148_loss_to_img = {}
label_149_loss_to_img = {}
label_150_loss_to_img = {}
label_151_loss_to_img = {}
label_152_loss_to_img = {}
label_153_loss_to_img = {}
label_154_loss_to_img = {}
label_155_loss_to_img = {}
label_156_loss_to_img = {}
label_157_loss_to_img = {}
label_158_loss_to_img = {}
label_159_loss_to_img = {}
label_160_loss_to_img = {}
label_161_loss_to_img = {}
label_162_loss_to_img = {}
label_163_loss_to_img = {}
label_164_loss_to_img = {}
label_165_loss_to_img = {}
label_166_loss_to_img = {}
label_167_loss_to_img = {}
label_168_loss_to_img = {}
label_169_loss_to_img = {}
label_170_loss_to_img = {}
label_171_loss_to_img = {}
label_172_loss_to_img = {}
label_173_loss_to_img = {}
label_174_loss_to_img = {}
label_175_loss_to_img = {}
label_176_loss_to_img = {}
label_177_loss_to_img = {}
label_178_loss_to_img = {}
label_179_loss_to_img = {}
label_180_loss_to_img = {}
label_181_loss_to_img = {}
label_182_loss_to_img = {}
label_183_loss_to_img = {}
label_184_loss_to_img = {}
label_185_loss_to_img = {}
label_186_loss_to_img = {}
label_187_loss_to_img = {}
label_188_loss_to_img = {}
label_189_loss_to_img = {}
label_190_loss_to_img = {}
label_191_loss_to_img = {}
label_192_loss_to_img = {}
label_193_loss_to_img = {}
label_194_loss_to_img = {}
label_195_loss_to_img = {}
label_196_loss_to_img = {}
label_197_loss_to_img = {}
label_198_loss_to_img = {}
label_199_loss_to_img = {}
label_200_loss_to_img = {}
label_201_loss_to_img = {}
label_202_loss_to_img = {}
label_203_loss_to_img = {}
label_204_loss_to_img = {}
label_205_loss_to_img = {}
label_206_loss_to_img = {}
label_207_loss_to_img = {}
label_208_loss_to_img = {}
label_209_loss_to_img = {}
label_210_loss_to_img = {}
label_211_loss_to_img = {}
label_212_loss_to_img = {}
label_213_loss_to_img = {}
label_214_loss_to_img = {}
label_215_loss_to_img = {}
label_216_loss_to_img = {}
label_217_loss_to_img = {}
label_218_loss_to_img = {}
label_219_loss_to_img = {}
label_220_loss_to_img = {}
label_221_loss_to_img = {}
label_222_loss_to_img = {}
label_223_loss_to_img = {}
label_224_loss_to_img = {}
label_225_loss_to_img = {}
label_226_loss_to_img = {}
label_227_loss_to_img = {}
label_228_loss_to_img = {}
label_229_loss_to_img = {}
label_230_loss_to_img = {}
label_231_loss_to_img = {}
label_232_loss_to_img = {}
label_233_loss_to_img = {}
label_234_loss_to_img = {}
label_235_loss_to_img = {}
label_236_loss_to_img = {}
label_237_loss_to_img = {}
label_238_loss_to_img = {}
label_239_loss_to_img = {}
label_240_loss_to_img = {}
label_241_loss_to_img = {}
label_242_loss_to_img = {}
label_243_loss_to_img = {}
label_244_loss_to_img = {}
label_245_loss_to_img = {}
label_246_loss_to_img = {}
label_247_loss_to_img = {}
label_248_loss_to_img = {}
label_249_loss_to_img = {}
label_250_loss_to_img = {}
label_251_loss_to_img = {}
label_252_loss_to_img = {}
label_253_loss_to_img = {}
label_254_loss_to_img = {}
label_255_loss_to_img = {}
label_256_loss_to_img = {}

with open("./results/spld_loss/alexnet_train_loss_file.txt", mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        img, loss = line.split(":")
        loss = float(loss)
        label = int(img.split("/")[4])

        if label == 0:
            if loss in label_0_loss_to_img.keys():
                label_0_loss_to_img[loss].append(img)
            else:
                label_0_loss_to_img[loss] = [img]
        if label == 1:
            if loss in label_1_loss_to_img.keys():
                label_1_loss_to_img[loss].append(img)
            else:
                label_1_loss_to_img[loss] = [img]
        if label == 2:
            if loss in label_2_loss_to_img.keys():
                label_2_loss_to_img[loss].append(img)
            else:
                label_2_loss_to_img[loss] = [img]
        if label == 3:
            if loss in label_3_loss_to_img.keys():
                label_3_loss_to_img[loss].append(img)
            else:
                label_3_loss_to_img[loss] = [img]
        if label == 4:
            if loss in label_4_loss_to_img.keys():
                label_4_loss_to_img[loss].append(img)
            else:
                label_4_loss_to_img[loss] = [img]
        if label == 5:
            if loss in label_5_loss_to_img.keys():
                label_5_loss_to_img[loss].append(img)
            else:
                label_5_loss_to_img[loss] = [img]
        if label == 6:
            if loss in label_6_loss_to_img.keys():
                label_6_loss_to_img[loss].append(img)
            else:
                label_6_loss_to_img[loss] = [img]
        if label == 7:
            if loss in label_7_loss_to_img.keys():
                label_7_loss_to_img[loss].append(img)
            else:
                label_7_loss_to_img[loss] = [img]
        if label == 8:
            if loss in label_8_loss_to_img.keys():
                label_8_loss_to_img[loss].append(img)
            else:
                label_8_loss_to_img[loss] = [img]
        if label == 9:
            if loss in label_9_loss_to_img.keys():
                label_9_loss_to_img[loss].append(img)
            else:
                label_9_loss_to_img[loss] = [img]

        if label == 10:
            if loss in label_10_loss_to_img.keys():
                label_10_loss_to_img[loss].append(img)
            else:
                label_10_loss_to_img[loss] = [img]
        if label == 11:
            if loss in label_11_loss_to_img.keys():
                label_11_loss_to_img[loss].append(img)
            else:
                label_11_loss_to_img[loss] = [img]
        if label == 12:
            if loss in label_12_loss_to_img.keys():
                label_12_loss_to_img[loss].append(img)
            else:
                label_12_loss_to_img[loss] = [img]
        if label == 13:
            if loss in label_13_loss_to_img.keys():
                label_13_loss_to_img[loss].append(img)
            else:
                label_13_loss_to_img[loss] = [img]
        if label == 14:
            if loss in label_14_loss_to_img.keys():
                label_14_loss_to_img[loss].append(img)
            else:
                label_14_loss_to_img[loss] = [img]
        if label == 15:
            if loss in label_15_loss_to_img.keys():
                label_15_loss_to_img[loss].append(img)
            else:
                label_15_loss_to_img[loss] = [img]
        if label == 16:
            if loss in label_16_loss_to_img.keys():
                label_16_loss_to_img[loss].append(img)
            else:
                label_16_loss_to_img[loss] = [img]
        if label == 17:
            if loss in label_17_loss_to_img.keys():
                label_17_loss_to_img[loss].append(img)
            else:
                label_17_loss_to_img[loss] = [img]
        if label == 18:
            if loss in label_18_loss_to_img.keys():
                label_18_loss_to_img[loss].append(img)
            else:
                label_18_loss_to_img[loss] = [img]
        if label == 19:
            if loss in label_19_loss_to_img.keys():
                label_19_loss_to_img[loss].append(img)
            else:
                label_19_loss_to_img[loss] = [img]
                
                
        if label == 20:
            if loss in label_0_loss_to_img.keys():
                label_20_loss_to_img[loss].append(img)
            else:
                label_20_loss_to_img[loss] = [img]
        if label == 21:
            if loss in label_21_loss_to_img.keys():
                label_21_loss_to_img[loss].append(img)
            else:
                label_21_loss_to_img[loss] = [img]
        if label == 22:
            if loss in label_22_loss_to_img.keys():
                label_22_loss_to_img[loss].append(img)
            else:
                label_22_loss_to_img[loss] = [img]
        if label == 23:
            if loss in label_23_loss_to_img.keys():
                label_23_loss_to_img[loss].append(img)
            else:
                label_23_loss_to_img[loss] = [img]
        if label == 24:
            if loss in label_24_loss_to_img.keys():
                label_24_loss_to_img[loss].append(img)
            else:
                label_24_loss_to_img[loss] = [img]
        if label == 25:
            if loss in label_25_loss_to_img.keys():
                label_25_loss_to_img[loss].append(img)
            else:
                label_25_loss_to_img[loss] = [img]
        if label == 26:
            if loss in label_26_loss_to_img.keys():
                label_26_loss_to_img[loss].append(img)
            else:
                label_26_loss_to_img[loss] = [img]
        if label == 27:
            if loss in label_27_loss_to_img.keys():
                label_27_loss_to_img[loss].append(img)
            else:
                label_27_loss_to_img[loss] = [img]
        if label == 28:
            if loss in label_28_loss_to_img.keys():
                label_28_loss_to_img[loss].append(img)
            else:
                label_28_loss_to_img[loss] = [img]
        if label == 29:
            if loss in label_29_loss_to_img.keys():
                label_29_loss_to_img[loss].append(img)
            else:
                label_29_loss_to_img[loss] = [img]

        if label == 30:
            if loss in label_30_loss_to_img.keys():
                label_30_loss_to_img[loss].append(img)
            else:
                label_30_loss_to_img[loss] = [img]
        if label == 31:
            if loss in label_31_loss_to_img.keys():
                label_31_loss_to_img[loss].append(img)
            else:
                label_31_loss_to_img[loss] = [img]
        if label == 32:
            if loss in label_32_loss_to_img.keys():
                label_32_loss_to_img[loss].append(img)
            else:
                label_32_loss_to_img[loss] = [img]
        if label == 33:
            if loss in label_33_loss_to_img.keys():
                label_33_loss_to_img[loss].append(img)
            else:
                label_33_loss_to_img[loss] = [img]
        if label == 34:
            if loss in label_34_loss_to_img.keys():
                label_34_loss_to_img[loss].append(img)
            else:
                label_34_loss_to_img[loss] = [img]
        if label == 35:
            if loss in label_35_loss_to_img.keys():
                label_35_loss_to_img[loss].append(img)
            else:
                label_35_loss_to_img[loss] = [img]
        if label == 36:
            if loss in label_36_loss_to_img.keys():
                label_36_loss_to_img[loss].append(img)
            else:
                label_36_loss_to_img[loss] = [img]
        if label == 37:
            if loss in label_37_loss_to_img.keys():
                label_37_loss_to_img[loss].append(img)
            else:
                label_37_loss_to_img[loss] = [img]
        if label == 38:
            if loss in label_38_loss_to_img.keys():
                label_38_loss_to_img[loss].append(img)
            else:
                label_38_loss_to_img[loss] = [img]
        if label == 39:
            if loss in label_39_loss_to_img.keys():
                label_39_loss_to_img[loss].append(img)
            else:
                label_39_loss_to_img[loss] = [img]
        
        
        if label == 40:
            if loss in label_40_loss_to_img.keys():
                label_40_loss_to_img[loss].append(img)
            else:
                label_40_loss_to_img[loss] = [img]
        if label == 41:
            if loss in label_41_loss_to_img.keys():
                label_41_loss_to_img[loss].append(img)
            else:
                label_41_loss_to_img[loss] = [img]
        if label == 42:
            if loss in label_42_loss_to_img.keys():
                label_42_loss_to_img[loss].append(img)
            else:
                label_42_loss_to_img[loss] = [img]
        if label == 43:
            if loss in label_43_loss_to_img.keys():
                label_43_loss_to_img[loss].append(img)
            else:
                label_43_loss_to_img[loss] = [img]
        if label == 44:
            if loss in label_44_loss_to_img.keys():
                label_44_loss_to_img[loss].append(img)
            else:
                label_44_loss_to_img[loss] = [img]
        if label == 45:
            if loss in label_45_loss_to_img.keys():
                label_45_loss_to_img[loss].append(img)
            else:
                label_45_loss_to_img[loss] = [img]
        if label == 46:
            if loss in label_46_loss_to_img.keys():
                label_46_loss_to_img[loss].append(img)
            else:
                label_46_loss_to_img[loss] = [img]
        if label == 47:
            if loss in label_47_loss_to_img.keys():
                label_47_loss_to_img[loss].append(img)
            else:
                label_47_loss_to_img[loss] = [img]
        if label == 48:
            if loss in label_48_loss_to_img.keys():
                label_48_loss_to_img[loss].append(img)
            else:
                label_48_loss_to_img[loss] = [img]
        if label == 49:
            if loss in label_49_loss_to_img.keys():
                label_49_loss_to_img[loss].append(img)
            else:
                label_49_loss_to_img[loss] = [img]

        if label == 50:
            if loss in label_50_loss_to_img.keys():
                label_50_loss_to_img[loss].append(img)
            else:
                label_50_loss_to_img[loss] = [img]
        if label == 51:
            if loss in label_51_loss_to_img.keys():
                label_51_loss_to_img[loss].append(img)
            else:
                label_51_loss_to_img[loss] = [img]
        if label == 52:
            if loss in label_52_loss_to_img.keys():
                label_52_loss_to_img[loss].append(img)
            else:
                label_52_loss_to_img[loss] = [img]
        if label == 53:
            if loss in label_53_loss_to_img.keys():
                label_53_loss_to_img[loss].append(img)
            else:
                label_53_loss_to_img[loss] = [img]
        if label == 54:
            if loss in label_54_loss_to_img.keys():
                label_54_loss_to_img[loss].append(img)
            else:
                label_54_loss_to_img[loss] = [img]
        if label == 55:
            if loss in label_55_loss_to_img.keys():
                label_55_loss_to_img[loss].append(img)
            else:
                label_55_loss_to_img[loss] = [img]
        if label == 56:
            if loss in label_56_loss_to_img.keys():
                label_56_loss_to_img[loss].append(img)
            else:
                label_56_loss_to_img[loss] = [img]
        if label == 57:
            if loss in label_57_loss_to_img.keys():
                label_57_loss_to_img[loss].append(img)
            else:
                label_57_loss_to_img[loss] = [img]
        if label == 58:
            if loss in label_58_loss_to_img.keys():
                label_58_loss_to_img[loss].append(img)
            else:
                label_58_loss_to_img[loss] = [img]
        if label == 59:
            if loss in label_59_loss_to_img.keys():
                label_59_loss_to_img[loss].append(img)
            else:
                label_59_loss_to_img[loss] = [img]
        
        
        if label == 60:
            if loss in label_60_loss_to_img.keys():
                label_60_loss_to_img[loss].append(img)
            else:
                label_60_loss_to_img[loss] = [img]
        if label == 61:
            if loss in label_61_loss_to_img.keys():
                label_61_loss_to_img[loss].append(img)
            else:
                label_61_loss_to_img[loss] = [img]
        if label == 62:
            if loss in label_62_loss_to_img.keys():
                label_62_loss_to_img[loss].append(img)
            else:
                label_62_loss_to_img[loss] = [img]
        if label == 63:
            if loss in label_63_loss_to_img.keys():
                label_63_loss_to_img[loss].append(img)
            else:
                label_63_loss_to_img[loss] = [img]
        if label == 64:
            if loss in label_64_loss_to_img.keys():
                label_64_loss_to_img[loss].append(img)
            else:
                label_64_loss_to_img[loss] = [img]
        if label == 65:
            if loss in label_65_loss_to_img.keys():
                label_65_loss_to_img[loss].append(img)
            else:
                label_65_loss_to_img[loss] = [img]
        if label == 66:
            if loss in label_66_loss_to_img.keys():
                label_66_loss_to_img[loss].append(img)
            else:
                label_66_loss_to_img[loss] = [img]
        if label == 67:
            if loss in label_67_loss_to_img.keys():
                label_67_loss_to_img[loss].append(img)
            else:
                label_67_loss_to_img[loss] = [img]
        if label == 68:
            if loss in label_68_loss_to_img.keys():
                label_68_loss_to_img[loss].append(img)
            else:
                label_68_loss_to_img[loss] = [img]
        if label == 69:
            if loss in label_69_loss_to_img.keys():
                label_69_loss_to_img[loss].append(img)
            else:
                label_69_loss_to_img[loss] = [img]

        if label == 70:
            if loss in label_70_loss_to_img.keys():
                label_70_loss_to_img[loss].append(img)
            else:
                label_70_loss_to_img[loss] = [img]
        if label == 71:
            if loss in label_71_loss_to_img.keys():
                label_71_loss_to_img[loss].append(img)
            else:
                label_71_loss_to_img[loss] = [img]
        if label == 72:
            if loss in label_72_loss_to_img.keys():
                label_72_loss_to_img[loss].append(img)
            else:
                label_72_loss_to_img[loss] = [img]
        if label == 73:
            if loss in label_73_loss_to_img.keys():
                label_73_loss_to_img[loss].append(img)
            else:
                label_73_loss_to_img[loss] = [img]
        if label == 74:
            if loss in label_74_loss_to_img.keys():
                label_74_loss_to_img[loss].append(img)
            else:
                label_74_loss_to_img[loss] = [img]
        if label == 75:
            if loss in label_75_loss_to_img.keys():
                label_75_loss_to_img[loss].append(img)
            else:
                label_75_loss_to_img[loss] = [img]
        if label == 76:
            if loss in label_76_loss_to_img.keys():
                label_76_loss_to_img[loss].append(img)
            else:
                label_76_loss_to_img[loss] = [img]
        if label == 77:
            if loss in label_77_loss_to_img.keys():
                label_77_loss_to_img[loss].append(img)
            else:
                label_77_loss_to_img[loss] = [img]
        if label == 78:
            if loss in label_78_loss_to_img.keys():
                label_78_loss_to_img[loss].append(img)
            else:
                label_78_loss_to_img[loss] = [img]
        if label == 79:
            if loss in label_79_loss_to_img.keys():
                label_79_loss_to_img[loss].append(img)
            else:
                label_79_loss_to_img[loss] = [img]
        
        
        if label == 80:
            if loss in label_80_loss_to_img.keys():
                label_80_loss_to_img[loss].append(img)
            else:
                label_80_loss_to_img[loss] = [img]
        if label == 81:
            if loss in label_81_loss_to_img.keys():
                label_81_loss_to_img[loss].append(img)
            else:
                label_81_loss_to_img[loss] = [img]
        if label == 82:
            if loss in label_82_loss_to_img.keys():
                label_82_loss_to_img[loss].append(img)
            else:
                label_82_loss_to_img[loss] = [img]
        if label == 83:
            if loss in label_83_loss_to_img.keys():
                label_83_loss_to_img[loss].append(img)
            else:
                label_83_loss_to_img[loss] = [img]
        if label == 84:
            if loss in label_84_loss_to_img.keys():
                label_84_loss_to_img[loss].append(img)
            else:
                label_84_loss_to_img[loss] = [img]
        if label == 85:
            if loss in label_85_loss_to_img.keys():
                label_85_loss_to_img[loss].append(img)
            else:
                label_85_loss_to_img[loss] = [img]
        if label == 86:
            if loss in label_86_loss_to_img.keys():
                label_86_loss_to_img[loss].append(img)
            else:
                label_86_loss_to_img[loss] = [img]
        if label == 87:
            if loss in label_87_loss_to_img.keys():
                label_87_loss_to_img[loss].append(img)
            else:
                label_87_loss_to_img[loss] = [img]
        if label == 88:
            if loss in label_88_loss_to_img.keys():
                label_88_loss_to_img[loss].append(img)
            else:
                label_88_loss_to_img[loss] = [img]
        if label == 89:
            if loss in label_89_loss_to_img.keys():
                label_89_loss_to_img[loss].append(img)
            else:
                label_89_loss_to_img[loss] = [img]

        if label == 90:
            if loss in label_90_loss_to_img.keys():
                label_90_loss_to_img[loss].append(img)
            else:
                label_90_loss_to_img[loss] = [img]
        if label == 91:
            if loss in label_91_loss_to_img.keys():
                label_91_loss_to_img[loss].append(img)
            else:
                label_91_loss_to_img[loss] = [img]
        if label == 92:
            if loss in label_92_loss_to_img.keys():
                label_92_loss_to_img[loss].append(img)
            else:
                label_92_loss_to_img[loss] = [img]
        if label == 93:
            if loss in label_93_loss_to_img.keys():
                label_93_loss_to_img[loss].append(img)
            else:
                label_93_loss_to_img[loss] = [img]
        if label == 94:
            if loss in label_94_loss_to_img.keys():
                label_94_loss_to_img[loss].append(img)
            else:
                label_94_loss_to_img[loss] = [img]
        if label == 95:
            if loss in label_95_loss_to_img.keys():
                label_95_loss_to_img[loss].append(img)
            else:
                label_95_loss_to_img[loss] = [img]
        if label == 96:
            if loss in label_96_loss_to_img.keys():
                label_96_loss_to_img[loss].append(img)
            else:
                label_96_loss_to_img[loss] = [img]
        if label == 97:
            if loss in label_97_loss_to_img.keys():
                label_97_loss_to_img[loss].append(img)
            else:
                label_97_loss_to_img[loss] = [img]
        if label == 98:
            if loss in label_98_loss_to_img.keys():
                label_98_loss_to_img[loss].append(img)
            else:
                label_98_loss_to_img[loss] = [img]
        if label == 99:
            if loss in label_99_loss_to_img.keys():
                label_99_loss_to_img[loss].append(img)
            else:
                label_99_loss_to_img[loss] = [img]
        
        

        if label == 100:
            if loss in label_100_loss_to_img.keys():
                label_100_loss_to_img[loss].append(img)
            else:
                label_100_loss_to_img[loss] = [img]
        if label == 101:
            if loss in label_101_loss_to_img.keys():
                label_101_loss_to_img[loss].append(img)
            else:
                label_101_loss_to_img[loss] = [img]
        if label == 102:
            if loss in label_102_loss_to_img.keys():
                label_102_loss_to_img[loss].append(img)
            else:
                label_102_loss_to_img[loss] = [img]
        if label == 103:
            if loss in label_103_loss_to_img.keys():
                label_103_loss_to_img[loss].append(img)
            else:
                label_103_loss_to_img[loss] = [img]
        if label == 104:
            if loss in label_104_loss_to_img.keys():
                label_104_loss_to_img[loss].append(img)
            else:
                label_104_loss_to_img[loss] = [img]
        if label == 105:
            if loss in label_105_loss_to_img.keys():
                label_105_loss_to_img[loss].append(img)
            else:
                label_105_loss_to_img[loss] = [img]
        if label == 106:
            if loss in label_106_loss_to_img.keys():
                label_106_loss_to_img[loss].append(img)
            else:
                label_106_loss_to_img[loss] = [img]
        if label == 107:
            if loss in label_107_loss_to_img.keys():
                label_107_loss_to_img[loss].append(img)
            else:
                label_107_loss_to_img[loss] = [img]
        if label == 108:
            if loss in label_108_loss_to_img.keys():
                label_108_loss_to_img[loss].append(img)
            else:
                label_108_loss_to_img[loss] = [img]
        if label == 109:
            if loss in label_109_loss_to_img.keys():
                label_109_loss_to_img[loss].append(img)
            else:
                label_109_loss_to_img[loss] = [img]
        
        
        if label == 110:
            if loss in label_110_loss_to_img.keys():
                label_110_loss_to_img[loss].append(img)
            else:
                label_110_loss_to_img[loss] = [img]
        if label == 111:
            if loss in label_111_loss_to_img.keys():
                label_111_loss_to_img[loss].append(img)
            else:
                label_111_loss_to_img[loss] = [img]
        if label == 112:
            if loss in label_112_loss_to_img.keys():
                label_112_loss_to_img[loss].append(img)
            else:
                label_112_loss_to_img[loss] = [img]
        if label == 113:
            if loss in label_113_loss_to_img.keys():
                label_113_loss_to_img[loss].append(img)
            else:
                label_113_loss_to_img[loss] = [img]
        if label == 114:
            if loss in label_114_loss_to_img.keys():
                label_114_loss_to_img[loss].append(img)
            else:
                label_114_loss_to_img[loss] = [img]
        if label == 115:
            if loss in label_115_loss_to_img.keys():
                label_115_loss_to_img[loss].append(img)
            else:
                label_115_loss_to_img[loss] = [img]
        if label == 116:
            if loss in label_116_loss_to_img.keys():
                label_116_loss_to_img[loss].append(img)
            else:
                label_116_loss_to_img[loss] = [img]
        if label == 117:
            if loss in label_117_loss_to_img.keys():
                label_117_loss_to_img[loss].append(img)
            else:
                label_117_loss_to_img[loss] = [img]
        if label == 118:
            if loss in label_118_loss_to_img.keys():
                label_118_loss_to_img[loss].append(img)
            else:
                label_118_loss_to_img[loss] = [img]
        if label == 119:
            if loss in label_119_loss_to_img.keys():
                label_119_loss_to_img[loss].append(img)
            else:
                label_119_loss_to_img[loss] = [img]

        if label == 120:
            if loss in label_120_loss_to_img.keys():
                label_120_loss_to_img[loss].append(img)
            else:
                label_120_loss_to_img[loss] = [img]
        if label == 121:
            if loss in label_121_loss_to_img.keys():
                label_121_loss_to_img[loss].append(img)
            else:
                label_121_loss_to_img[loss] = [img]
        if label == 122:
            if loss in label_122_loss_to_img.keys():
                label_122_loss_to_img[loss].append(img)
            else:
                label_122_loss_to_img[loss] = [img]
        if label == 123:
            if loss in label_123_loss_to_img.keys():
                label_123_loss_to_img[loss].append(img)
            else:
                label_123_loss_to_img[loss] = [img]
        if label == 124:
            if loss in label_124_loss_to_img.keys():
                label_124_loss_to_img[loss].append(img)
            else:
                label_124_loss_to_img[loss] = [img]
        if label == 125:
            if loss in label_125_loss_to_img.keys():
                label_125_loss_to_img[loss].append(img)
            else:
                label_125_loss_to_img[loss] = [img]
        if label == 126:
            if loss in label_126_loss_to_img.keys():
                label_126_loss_to_img[loss].append(img)
            else:
                label_126_loss_to_img[loss] = [img]
        if label == 127:
            if loss in label_127_loss_to_img.keys():
                label_127_loss_to_img[loss].append(img)
            else:
                label_127_loss_to_img[loss] = [img]
        if label == 128:
            if loss in label_128_loss_to_img.keys():
                label_128_loss_to_img[loss].append(img)
            else:
                label_128_loss_to_img[loss] = [img]
        if label == 129:
            if loss in label_129_loss_to_img.keys():
                label_129_loss_to_img[loss].append(img)
            else:
                label_129_loss_to_img[loss] = [img]
        
        
        
        if label == 130:
            if loss in label_130_loss_to_img.keys():
                label_130_loss_to_img[loss].append(img)
            else:
                label_130_loss_to_img[loss] = [img]
        if label == 131:
            if loss in label_131_loss_to_img.keys():
                label_131_loss_to_img[loss].append(img)
            else:
                label_131_loss_to_img[loss] = [img]
        if label == 132:
            if loss in label_132_loss_to_img.keys():
                label_132_loss_to_img[loss].append(img)
            else:
                label_132_loss_to_img[loss] = [img]
        if label == 133:
            if loss in label_133_loss_to_img.keys():
                label_133_loss_to_img[loss].append(img)
            else:
                label_133_loss_to_img[loss] = [img]
        if label == 134:
            if loss in label_134_loss_to_img.keys():
                label_134_loss_to_img[loss].append(img)
            else:
                label_134_loss_to_img[loss] = [img]
        if label == 135:
            if loss in label_135_loss_to_img.keys():
                label_135_loss_to_img[loss].append(img)
            else:
                label_135_loss_to_img[loss] = [img]
        if label == 136:
            if loss in label_136_loss_to_img.keys():
                label_136_loss_to_img[loss].append(img)
            else:
                label_136_loss_to_img[loss] = [img]
        if label == 137:
            if loss in label_137_loss_to_img.keys():
                label_137_loss_to_img[loss].append(img)
            else:
                label_137_loss_to_img[loss] = [img]
        if label == 138:
            if loss in label_138_loss_to_img.keys():
                label_138_loss_to_img[loss].append(img)
            else:
                label_138_loss_to_img[loss] = [img]
        if label == 139:
            if loss in label_139_loss_to_img.keys():
                label_139_loss_to_img[loss].append(img)
            else:
                label_139_loss_to_img[loss] = [img]

        if label == 140:
            if loss in label_140_loss_to_img.keys():
                label_140_loss_to_img[loss].append(img)
            else:
                label_140_loss_to_img[loss] = [img]
        if label == 141:
            if loss in label_141_loss_to_img.keys():
                label_141_loss_to_img[loss].append(img)
            else:
                label_141_loss_to_img[loss] = [img]
        if label == 142:
            if loss in label_142_loss_to_img.keys():
                label_142_loss_to_img[loss].append(img)
            else:
                label_142_loss_to_img[loss] = [img]
        if label == 143:
            if loss in label_143_loss_to_img.keys():
                label_143_loss_to_img[loss].append(img)
            else:
                label_143_loss_to_img[loss] = [img]
        if label == 144:
            if loss in label_144_loss_to_img.keys():
                label_144_loss_to_img[loss].append(img)
            else:
                label_144_loss_to_img[loss] = [img]
        if label == 145:
            if loss in label_145_loss_to_img.keys():
                label_145_loss_to_img[loss].append(img)
            else:
                label_145_loss_to_img[loss] = [img]
        if label == 146:
            if loss in label_146_loss_to_img.keys():
                label_146_loss_to_img[loss].append(img)
            else:
                label_146_loss_to_img[loss] = [img]
        if label == 147:
            if loss in label_147_loss_to_img.keys():
                label_147_loss_to_img[loss].append(img)
            else:
                label_147_loss_to_img[loss] = [img]
        if label == 148:
            if loss in label_148_loss_to_img.keys():
                label_148_loss_to_img[loss].append(img)
            else:
                label_148_loss_to_img[loss] = [img]
        if label == 149:
            if loss in label_149_loss_to_img.keys():
                label_149_loss_to_img[loss].append(img)
            else:
                label_149_loss_to_img[loss] = [img]





        if label == 150:
            if loss in label_150_loss_to_img.keys():
                label_150_loss_to_img[loss].append(img)
            else:
                label_150_loss_to_img[loss] = [img]
        if label == 151:
            if loss in label_151_loss_to_img.keys():
                label_151_loss_to_img[loss].append(img)
            else:
                label_151_loss_to_img[loss] = [img]
        if label == 152:
            if loss in label_152_loss_to_img.keys():
                label_152_loss_to_img[loss].append(img)
            else:
                label_152_loss_to_img[loss] = [img]
        if label == 153:
            if loss in label_153_loss_to_img.keys():
                label_153_loss_to_img[loss].append(img)
            else:
                label_153_loss_to_img[loss] = [img]
        if label == 154:
            if loss in label_154_loss_to_img.keys():
                label_154_loss_to_img[loss].append(img)
            else:
                label_154_loss_to_img[loss] = [img]
        if label == 155:
            if loss in label_155_loss_to_img.keys():
                label_155_loss_to_img[loss].append(img)
            else:
                label_155_loss_to_img[loss] = [img]
        if label == 156:
            if loss in label_156_loss_to_img.keys():
                label_156_loss_to_img[loss].append(img)
            else:
                label_156_loss_to_img[loss] = [img]
        if label == 157:
            if loss in label_157_loss_to_img.keys():
                label_157_loss_to_img[loss].append(img)
            else:
                label_157_loss_to_img[loss] = [img]
        if label == 158:
            if loss in label_158_loss_to_img.keys():
                label_158_loss_to_img[loss].append(img)
            else:
                label_158_loss_to_img[loss] = [img]
        if label == 159:
            if loss in label_159_loss_to_img.keys():
                label_159_loss_to_img[loss].append(img)
            else:
                label_159_loss_to_img[loss] = [img]

        if label == 160:
            if loss in label_160_loss_to_img.keys():
                label_160_loss_to_img[loss].append(img)
            else:
                label_160_loss_to_img[loss] = [img]
        if label == 161:
            if loss in label_161_loss_to_img.keys():
                label_161_loss_to_img[loss].append(img)
            else:
                label_161_loss_to_img[loss] = [img]
        if label == 162:
            if loss in label_162_loss_to_img.keys():
                label_162_loss_to_img[loss].append(img)
            else:
                label_162_loss_to_img[loss] = [img]
        if label == 163:
            if loss in label_163_loss_to_img.keys():
                label_163_loss_to_img[loss].append(img)
            else:
                label_163_loss_to_img[loss] = [img]
        if label == 164:
            if loss in label_164_loss_to_img.keys():
                label_164_loss_to_img[loss].append(img)
            else:
                label_164_loss_to_img[loss] = [img]
        if label == 165:
            if loss in label_165_loss_to_img.keys():
                label_165_loss_to_img[loss].append(img)
            else:
                label_165_loss_to_img[loss] = [img]
        if label == 166:
            if loss in label_166_loss_to_img.keys():
                label_166_loss_to_img[loss].append(img)
            else:
                label_166_loss_to_img[loss] = [img]
        if label == 167:
            if loss in label_167_loss_to_img.keys():
                label_167_loss_to_img[loss].append(img)
            else:
                label_167_loss_to_img[loss] = [img]
        if label == 168:
            if loss in label_168_loss_to_img.keys():
                label_168_loss_to_img[loss].append(img)
            else:
                label_168_loss_to_img[loss] = [img]
        if label == 169:
            if loss in label_169_loss_to_img.keys():
                label_169_loss_to_img[loss].append(img)
            else:
                label_169_loss_to_img[loss] = [img]
        
        
        
        if label == 170:
            if loss in label_170_loss_to_img.keys():
                label_170_loss_to_img[loss].append(img)
            else:
                label_170_loss_to_img[loss] = [img]
        if label == 171:
            if loss in label_171_loss_to_img.keys():
                label_171_loss_to_img[loss].append(img)
            else:
                label_171_loss_to_img[loss] = [img]
        if label == 172:
            if loss in label_172_loss_to_img.keys():
                label_172_loss_to_img[loss].append(img)
            else:
                label_172_loss_to_img[loss] = [img]
        if label == 173:
            if loss in label_173_loss_to_img.keys():
                label_173_loss_to_img[loss].append(img)
            else:
                label_173_loss_to_img[loss] = [img]
        if label == 174:
            if loss in label_174_loss_to_img.keys():
                label_174_loss_to_img[loss].append(img)
            else:
                label_174_loss_to_img[loss] = [img]
        if label == 175:
            if loss in label_175_loss_to_img.keys():
                label_175_loss_to_img[loss].append(img)
            else:
                label_175_loss_to_img[loss] = [img]
        if label == 176:
            if loss in label_176_loss_to_img.keys():
                label_176_loss_to_img[loss].append(img)
            else:
                label_176_loss_to_img[loss] = [img]
        if label == 177:
            if loss in label_177_loss_to_img.keys():
                label_177_loss_to_img[loss].append(img)
            else:
                label_177_loss_to_img[loss] = [img]
        if label == 178:
            if loss in label_178_loss_to_img.keys():
                label_178_loss_to_img[loss].append(img)
            else:
                label_178_loss_to_img[loss] = [img]
        if label == 179:
            if loss in label_179_loss_to_img.keys():
                label_179_loss_to_img[loss].append(img)
            else:
                label_179_loss_to_img[loss] = [img]

        if label == 180:
            if loss in label_180_loss_to_img.keys():
                label_180_loss_to_img[loss].append(img)
            else:
                label_180_loss_to_img[loss] = [img]
        if label == 181:
            if loss in label_181_loss_to_img.keys():
                label_181_loss_to_img[loss].append(img)
            else:
                label_181_loss_to_img[loss] = [img]
        if label == 182:
            if loss in label_182_loss_to_img.keys():
                label_182_loss_to_img[loss].append(img)
            else:
                label_182_loss_to_img[loss] = [img]
        if label == 183:
            if loss in label_183_loss_to_img.keys():
                label_183_loss_to_img[loss].append(img)
            else:
                label_183_loss_to_img[loss] = [img]
        if label == 184:
            if loss in label_184_loss_to_img.keys():
                label_184_loss_to_img[loss].append(img)
            else:
                label_184_loss_to_img[loss] = [img]
        if label == 185:
            if loss in label_185_loss_to_img.keys():
                label_185_loss_to_img[loss].append(img)
            else:
                label_185_loss_to_img[loss] = [img]
        if label == 186:
            if loss in label_186_loss_to_img.keys():
                label_186_loss_to_img[loss].append(img)
            else:
                label_186_loss_to_img[loss] = [img]
        if label == 187:
            if loss in label_187_loss_to_img.keys():
                label_187_loss_to_img[loss].append(img)
            else:
                label_187_loss_to_img[loss] = [img]
        if label == 188:
            if loss in label_188_loss_to_img.keys():
                label_188_loss_to_img[loss].append(img)
            else:
                label_188_loss_to_img[loss] = [img]
        if label == 189:
            if loss in label_189_loss_to_img.keys():
                label_189_loss_to_img[loss].append(img)
            else:
                label_189_loss_to_img[loss] = [img]
                
        
        
        if label == 190:
            if loss in label_190_loss_to_img.keys():
                label_190_loss_to_img[loss].append(img)
            else:
                label_190_loss_to_img[loss] = [img]
        if label == 191:
            if loss in label_191_loss_to_img.keys():
                label_191_loss_to_img[loss].append(img)
            else:
                label_191_loss_to_img[loss] = [img]
        if label == 192:
            if loss in label_192_loss_to_img.keys():
                label_192_loss_to_img[loss].append(img)
            else:
                label_192_loss_to_img[loss] = [img]
        if label == 193:
            if loss in label_193_loss_to_img.keys():
                label_193_loss_to_img[loss].append(img)
            else:
                label_193_loss_to_img[loss] = [img]
        if label == 194:
            if loss in label_194_loss_to_img.keys():
                label_194_loss_to_img[loss].append(img)
            else:
                label_194_loss_to_img[loss] = [img]
        if label == 195:
            if loss in label_195_loss_to_img.keys():
                label_195_loss_to_img[loss].append(img)
            else:
                label_195_loss_to_img[loss] = [img]
        if label == 196:
            if loss in label_196_loss_to_img.keys():
                label_196_loss_to_img[loss].append(img)
            else:
                label_196_loss_to_img[loss] = [img]
        if label == 197:
            if loss in label_197_loss_to_img.keys():
                label_197_loss_to_img[loss].append(img)
            else:
                label_197_loss_to_img[loss] = [img]
        if label == 198:
            if loss in label_198_loss_to_img.keys():
                label_198_loss_to_img[loss].append(img)
            else:
                label_198_loss_to_img[loss] = [img]
        if label == 199:
            if loss in label_199_loss_to_img.keys():
                label_199_loss_to_img[loss].append(img)
            else:
                label_199_loss_to_img[loss] = [img]
        
        
        
        if label == 200:
            if loss in label_200_loss_to_img.keys():
                label_200_loss_to_img[loss].append(img)
            else:
                label_200_loss_to_img[loss] = [img]
        if label == 201:
            if loss in label_201_loss_to_img.keys():
                label_201_loss_to_img[loss].append(img)
            else:
                label_201_loss_to_img[loss] = [img]
        if label == 202:
            if loss in label_202_loss_to_img.keys():
                label_202_loss_to_img[loss].append(img)
            else:
                label_202_loss_to_img[loss] = [img]
        if label == 203:
            if loss in label_203_loss_to_img.keys():
                label_203_loss_to_img[loss].append(img)
            else:
                label_203_loss_to_img[loss] = [img]
        if label == 204:
            if loss in label_204_loss_to_img.keys():
                label_204_loss_to_img[loss].append(img)
            else:
                label_204_loss_to_img[loss] = [img]
        if label == 205:
            if loss in label_205_loss_to_img.keys():
                label_205_loss_to_img[loss].append(img)
            else:
                label_205_loss_to_img[loss] = [img]
        if label == 206:
            if loss in label_206_loss_to_img.keys():
                label_206_loss_to_img[loss].append(img)
            else:
                label_206_loss_to_img[loss] = [img]
        if label == 207:
            if loss in label_207_loss_to_img.keys():
                label_207_loss_to_img[loss].append(img)
            else:
                label_207_loss_to_img[loss] = [img]
        if label == 208:
            if loss in label_208_loss_to_img.keys():
                label_208_loss_to_img[loss].append(img)
            else:
                label_208_loss_to_img[loss] = [img]
        if label == 209:
            if loss in label_209_loss_to_img.keys():
                label_209_loss_to_img[loss].append(img)
            else:
                label_209_loss_to_img[loss] = [img]
        
        
        
        if label == 210:
            if loss in label_210_loss_to_img.keys():
                label_210_loss_to_img[loss].append(img)
            else:
                label_210_loss_to_img[loss] = [img]
        if label == 211:
            if loss in label_211_loss_to_img.keys():
                label_211_loss_to_img[loss].append(img)
            else:
                label_211_loss_to_img[loss] = [img]
        if label == 212:
            if loss in label_212_loss_to_img.keys():
                label_212_loss_to_img[loss].append(img)
            else:
                label_212_loss_to_img[loss] = [img]
        if label == 213:
            if loss in label_213_loss_to_img.keys():
                label_213_loss_to_img[loss].append(img)
            else:
                label_213_loss_to_img[loss] = [img]
        if label == 214:
            if loss in label_214_loss_to_img.keys():
                label_214_loss_to_img[loss].append(img)
            else:
                label_214_loss_to_img[loss] = [img]
        if label == 215:
            if loss in label_215_loss_to_img.keys():
                label_215_loss_to_img[loss].append(img)
            else:
                label_215_loss_to_img[loss] = [img]
        if label == 216:
            if loss in label_216_loss_to_img.keys():
                label_216_loss_to_img[loss].append(img)
            else:
                label_216_loss_to_img[loss] = [img]
        if label == 217:
            if loss in label_217_loss_to_img.keys():
                label_217_loss_to_img[loss].append(img)
            else:
                label_217_loss_to_img[loss] = [img]
        if label == 218:
            if loss in label_218_loss_to_img.keys():
                label_218_loss_to_img[loss].append(img)
            else:
                label_218_loss_to_img[loss] = [img]
        if label == 219:
            if loss in label_219_loss_to_img.keys():
                label_219_loss_to_img[loss].append(img)
            else:
                label_219_loss_to_img[loss] = [img]
        
        
        if label == 220:
            if loss in label_220_loss_to_img.keys():
                label_220_loss_to_img[loss].append(img)
            else:
                label_220_loss_to_img[loss] = [img]
        if label == 221:
            if loss in label_221_loss_to_img.keys():
                label_221_loss_to_img[loss].append(img)
            else:
                label_221_loss_to_img[loss] = [img]
        if label == 222:
            if loss in label_222_loss_to_img.keys():
                label_222_loss_to_img[loss].append(img)
            else:
                label_222_loss_to_img[loss] = [img]
        if label == 223:
            if loss in label_223_loss_to_img.keys():
                label_223_loss_to_img[loss].append(img)
            else:
                label_223_loss_to_img[loss] = [img]
        if label == 224:
            if loss in label_224_loss_to_img.keys():
                label_224_loss_to_img[loss].append(img)
            else:
                label_224_loss_to_img[loss] = [img]
        if label == 225:
            if loss in label_225_loss_to_img.keys():
                label_225_loss_to_img[loss].append(img)
            else:
                label_225_loss_to_img[loss] = [img]
        if label == 226:
            if loss in label_226_loss_to_img.keys():
                label_226_loss_to_img[loss].append(img)
            else:
                label_226_loss_to_img[loss] = [img]
        if label == 227:
            if loss in label_227_loss_to_img.keys():
                label_227_loss_to_img[loss].append(img)
            else:
                label_227_loss_to_img[loss] = [img]
        if label == 228:
            if loss in label_228_loss_to_img.keys():
                label_228_loss_to_img[loss].append(img)
            else:
                label_228_loss_to_img[loss] = [img]
        if label == 229:
            if loss in label_229_loss_to_img.keys():
                label_229_loss_to_img[loss].append(img)
            else:
                label_229_loss_to_img[loss] = [img]
        
        
        
        if label == 230:
            if loss in label_230_loss_to_img.keys():
                label_230_loss_to_img[loss].append(img)
            else:
                label_230_loss_to_img[loss] = [img]
        if label == 231:
            if loss in label_231_loss_to_img.keys():
                label_231_loss_to_img[loss].append(img)
            else:
                label_231_loss_to_img[loss] = [img]
        if label == 232:
            if loss in label_232_loss_to_img.keys():
                label_232_loss_to_img[loss].append(img)
            else:
                label_232_loss_to_img[loss] = [img]
        if label == 233:
            if loss in label_233_loss_to_img.keys():
                label_233_loss_to_img[loss].append(img)
            else:
                label_233_loss_to_img[loss] = [img]
        if label == 234:
            if loss in label_234_loss_to_img.keys():
                label_234_loss_to_img[loss].append(img)
            else:
                label_234_loss_to_img[loss] = [img]
        if label == 235:
            if loss in label_235_loss_to_img.keys():
                label_235_loss_to_img[loss].append(img)
            else:
                label_235_loss_to_img[loss] = [img]
        if label == 236:
            if loss in label_236_loss_to_img.keys():
                label_236_loss_to_img[loss].append(img)
            else:
                label_236_loss_to_img[loss] = [img]
        if label == 237:
            if loss in label_237_loss_to_img.keys():
                label_237_loss_to_img[loss].append(img)
            else:
                label_237_loss_to_img[loss] = [img]
        if label == 238:
            if loss in label_238_loss_to_img.keys():
                label_238_loss_to_img[loss].append(img)
            else:
                label_238_loss_to_img[loss] = [img]
        if label == 239:
            if loss in label_239_loss_to_img.keys():
                label_239_loss_to_img[loss].append(img)
            else:
                label_239_loss_to_img[loss] = [img]
        
        
        
        if label == 240:
            if loss in label_240_loss_to_img.keys():
                label_240_loss_to_img[loss].append(img)
            else:
                label_240_loss_to_img[loss] = [img]
        if label == 241:
            if loss in label_241_loss_to_img.keys():
                label_241_loss_to_img[loss].append(img)
            else:
                label_241_loss_to_img[loss] = [img]
        if label == 242:
            if loss in label_242_loss_to_img.keys():
                label_242_loss_to_img[loss].append(img)
            else:
                label_242_loss_to_img[loss] = [img]
        if label == 243:
            if loss in label_243_loss_to_img.keys():
                label_243_loss_to_img[loss].append(img)
            else:
                label_243_loss_to_img[loss] = [img]
        if label == 244:
            if loss in label_244_loss_to_img.keys():
                label_244_loss_to_img[loss].append(img)
            else:
                label_244_loss_to_img[loss] = [img]
        if label == 245:
            if loss in label_245_loss_to_img.keys():
                label_245_loss_to_img[loss].append(img)
            else:
                label_245_loss_to_img[loss] = [img]
        if label == 246:
            if loss in label_246_loss_to_img.keys():
                label_246_loss_to_img[loss].append(img)
            else:
                label_246_loss_to_img[loss] = [img]
        if label == 247:
            if loss in label_247_loss_to_img.keys():
                label_247_loss_to_img[loss].append(img)
            else:
                label_247_loss_to_img[loss] = [img]
        if label == 248:
            if loss in label_248_loss_to_img.keys():
                label_248_loss_to_img[loss].append(img)
            else:
                label_248_loss_to_img[loss] = [img]
        if label == 249:
            if loss in label_249_loss_to_img.keys():
                label_249_loss_to_img[loss].append(img)
            else:
                label_249_loss_to_img[loss] = [img]
        
        
        if label == 250:
            if loss in label_250_loss_to_img.keys():
                label_250_loss_to_img[loss].append(img)
            else:
                label_250_loss_to_img[loss] = [img]
        if label == 251:
            if loss in label_251_loss_to_img.keys():
                label_251_loss_to_img[loss].append(img)
            else:
                label_251_loss_to_img[loss] = [img]
        if label == 252:
            if loss in label_252_loss_to_img.keys():
                label_252_loss_to_img[loss].append(img)
            else:
                label_252_loss_to_img[loss] = [img]
        if label == 253:
            if loss in label_253_loss_to_img.keys():
                label_253_loss_to_img[loss].append(img)
            else:
                label_253_loss_to_img[loss] = [img]
        if label == 254:
            if loss in label_254_loss_to_img.keys():
                label_254_loss_to_img[loss].append(img)
            else:
                label_254_loss_to_img[loss] = [img]
        if label == 255:
            if loss in label_255_loss_to_img.keys():
                label_255_loss_to_img[loss].append(img)
            else:
                label_255_loss_to_img[loss] = [img]
        if label == 256:
            if loss in label_256_loss_to_img.keys():
                label_256_loss_to_img[loss].append(img)
            else:
                label_256_loss_to_img[loss] = [img]


label_0_loss = np.array(list(map(float, list(label_0_loss_to_img.keys())))).reshape(-1, 1)
label_1_loss = np.array(list(map(float, list(label_1_loss_to_img.keys())))).reshape(-1, 1)
label_2_loss = np.array(list(map(float, list(label_2_loss_to_img.keys())))).reshape(-1, 1)
label_3_loss = np.array(list(map(float, list(label_3_loss_to_img.keys())))).reshape(-1, 1)
label_4_loss = np.array(list(map(float, list(label_4_loss_to_img.keys())))).reshape(-1, 1)
label_5_loss = np.array(list(map(float, list(label_5_loss_to_img.keys())))).reshape(-1, 1)
label_6_loss = np.array(list(map(float, list(label_6_loss_to_img.keys())))).reshape(-1, 1)
label_7_loss = np.array(list(map(float, list(label_7_loss_to_img.keys())))).reshape(-1, 1)
label_8_loss = np.array(list(map(float, list(label_8_loss_to_img.keys())))).reshape(-1, 1)
label_9_loss = np.array(list(map(float, list(label_9_loss_to_img.keys())))).reshape(-1, 1)

label_10_loss = np.array(list(map(float, list(label_10_loss_to_img.keys())))).reshape(-1, 1)
label_11_loss = np.array(list(map(float, list(label_11_loss_to_img.keys())))).reshape(-1, 1)
label_12_loss = np.array(list(map(float, list(label_12_loss_to_img.keys())))).reshape(-1, 1)
label_13_loss = np.array(list(map(float, list(label_13_loss_to_img.keys())))).reshape(-1, 1)
label_14_loss = np.array(list(map(float, list(label_14_loss_to_img.keys())))).reshape(-1, 1)
label_15_loss = np.array(list(map(float, list(label_15_loss_to_img.keys())))).reshape(-1, 1)
label_16_loss = np.array(list(map(float, list(label_16_loss_to_img.keys())))).reshape(-1, 1)
label_17_loss = np.array(list(map(float, list(label_17_loss_to_img.keys())))).reshape(-1, 1)
label_18_loss = np.array(list(map(float, list(label_18_loss_to_img.keys())))).reshape(-1, 1)
label_19_loss = np.array(list(map(float, list(label_19_loss_to_img.keys())))).reshape(-1, 1)


label_20_loss = np.array(list(map(float, list(label_20_loss_to_img.keys())))).reshape(-1, 1)
label_21_loss = np.array(list(map(float, list(label_21_loss_to_img.keys())))).reshape(-1, 1)
label_22_loss = np.array(list(map(float, list(label_22_loss_to_img.keys())))).reshape(-1, 1)
label_23_loss = np.array(list(map(float, list(label_23_loss_to_img.keys())))).reshape(-1, 1)
label_24_loss = np.array(list(map(float, list(label_24_loss_to_img.keys())))).reshape(-1, 1)
label_25_loss = np.array(list(map(float, list(label_25_loss_to_img.keys())))).reshape(-1, 1)
label_26_loss = np.array(list(map(float, list(label_26_loss_to_img.keys())))).reshape(-1, 1)
label_27_loss = np.array(list(map(float, list(label_27_loss_to_img.keys())))).reshape(-1, 1)
label_28_loss = np.array(list(map(float, list(label_28_loss_to_img.keys())))).reshape(-1, 1)
label_29_loss = np.array(list(map(float, list(label_29_loss_to_img.keys())))).reshape(-1, 1)

label_30_loss = np.array(list(map(float, list(label_30_loss_to_img.keys())))).reshape(-1, 1)
label_31_loss = np.array(list(map(float, list(label_31_loss_to_img.keys())))).reshape(-1, 1)
label_32_loss = np.array(list(map(float, list(label_32_loss_to_img.keys())))).reshape(-1, 1)
label_33_loss = np.array(list(map(float, list(label_33_loss_to_img.keys())))).reshape(-1, 1)
label_34_loss = np.array(list(map(float, list(label_34_loss_to_img.keys())))).reshape(-1, 1)
label_35_loss = np.array(list(map(float, list(label_35_loss_to_img.keys())))).reshape(-1, 1)
label_36_loss = np.array(list(map(float, list(label_36_loss_to_img.keys())))).reshape(-1, 1)
label_37_loss = np.array(list(map(float, list(label_37_loss_to_img.keys())))).reshape(-1, 1)
label_38_loss = np.array(list(map(float, list(label_38_loss_to_img.keys())))).reshape(-1, 1)
label_39_loss = np.array(list(map(float, list(label_39_loss_to_img.keys())))).reshape(-1, 1)


label_40_loss = np.array(list(map(float, list(label_40_loss_to_img.keys())))).reshape(-1, 1)
label_41_loss = np.array(list(map(float, list(label_41_loss_to_img.keys())))).reshape(-1, 1)
label_42_loss = np.array(list(map(float, list(label_42_loss_to_img.keys())))).reshape(-1, 1)
label_43_loss = np.array(list(map(float, list(label_43_loss_to_img.keys())))).reshape(-1, 1)
label_44_loss = np.array(list(map(float, list(label_44_loss_to_img.keys())))).reshape(-1, 1)
label_45_loss = np.array(list(map(float, list(label_45_loss_to_img.keys())))).reshape(-1, 1)
label_46_loss = np.array(list(map(float, list(label_46_loss_to_img.keys())))).reshape(-1, 1)
label_47_loss = np.array(list(map(float, list(label_47_loss_to_img.keys())))).reshape(-1, 1)
label_48_loss = np.array(list(map(float, list(label_48_loss_to_img.keys())))).reshape(-1, 1)
label_49_loss = np.array(list(map(float, list(label_49_loss_to_img.keys())))).reshape(-1, 1)


label_50_loss = np.array(list(map(float, list(label_50_loss_to_img.keys())))).reshape(-1, 1)
label_51_loss = np.array(list(map(float, list(label_51_loss_to_img.keys())))).reshape(-1, 1)
label_52_loss = np.array(list(map(float, list(label_52_loss_to_img.keys())))).reshape(-1, 1)
label_53_loss = np.array(list(map(float, list(label_53_loss_to_img.keys())))).reshape(-1, 1)
label_54_loss = np.array(list(map(float, list(label_54_loss_to_img.keys())))).reshape(-1, 1)
label_55_loss = np.array(list(map(float, list(label_55_loss_to_img.keys())))).reshape(-1, 1)
label_56_loss = np.array(list(map(float, list(label_56_loss_to_img.keys())))).reshape(-1, 1)
label_57_loss = np.array(list(map(float, list(label_57_loss_to_img.keys())))).reshape(-1, 1)
label_58_loss = np.array(list(map(float, list(label_58_loss_to_img.keys())))).reshape(-1, 1)
label_59_loss = np.array(list(map(float, list(label_59_loss_to_img.keys())))).reshape(-1, 1)


label_60_loss = np.array(list(map(float, list(label_60_loss_to_img.keys())))).reshape(-1, 1)
label_61_loss = np.array(list(map(float, list(label_61_loss_to_img.keys())))).reshape(-1, 1)
label_62_loss = np.array(list(map(float, list(label_62_loss_to_img.keys())))).reshape(-1, 1)
label_63_loss = np.array(list(map(float, list(label_63_loss_to_img.keys())))).reshape(-1, 1)
label_64_loss = np.array(list(map(float, list(label_64_loss_to_img.keys())))).reshape(-1, 1)
label_65_loss = np.array(list(map(float, list(label_65_loss_to_img.keys())))).reshape(-1, 1)
label_66_loss = np.array(list(map(float, list(label_66_loss_to_img.keys())))).reshape(-1, 1)
label_67_loss = np.array(list(map(float, list(label_67_loss_to_img.keys())))).reshape(-1, 1)
label_68_loss = np.array(list(map(float, list(label_68_loss_to_img.keys())))).reshape(-1, 1)
label_69_loss = np.array(list(map(float, list(label_69_loss_to_img.keys())))).reshape(-1, 1)


label_70_loss = np.array(list(map(float, list(label_70_loss_to_img.keys())))).reshape(-1, 1)
label_71_loss = np.array(list(map(float, list(label_71_loss_to_img.keys())))).reshape(-1, 1)
label_72_loss = np.array(list(map(float, list(label_72_loss_to_img.keys())))).reshape(-1, 1)
label_73_loss = np.array(list(map(float, list(label_73_loss_to_img.keys())))).reshape(-1, 1)
label_74_loss = np.array(list(map(float, list(label_74_loss_to_img.keys())))).reshape(-1, 1)
label_75_loss = np.array(list(map(float, list(label_75_loss_to_img.keys())))).reshape(-1, 1)
label_76_loss = np.array(list(map(float, list(label_76_loss_to_img.keys())))).reshape(-1, 1)
label_77_loss = np.array(list(map(float, list(label_77_loss_to_img.keys())))).reshape(-1, 1)
label_78_loss = np.array(list(map(float, list(label_78_loss_to_img.keys())))).reshape(-1, 1)
label_79_loss = np.array(list(map(float, list(label_79_loss_to_img.keys())))).reshape(-1, 1)

label_80_loss = np.array(list(map(float, list(label_80_loss_to_img.keys())))).reshape(-1, 1)
label_81_loss = np.array(list(map(float, list(label_81_loss_to_img.keys())))).reshape(-1, 1)
label_82_loss = np.array(list(map(float, list(label_82_loss_to_img.keys())))).reshape(-1, 1)
label_83_loss = np.array(list(map(float, list(label_83_loss_to_img.keys())))).reshape(-1, 1)
label_84_loss = np.array(list(map(float, list(label_84_loss_to_img.keys())))).reshape(-1, 1)
label_85_loss = np.array(list(map(float, list(label_85_loss_to_img.keys())))).reshape(-1, 1)
label_86_loss = np.array(list(map(float, list(label_86_loss_to_img.keys())))).reshape(-1, 1)
label_87_loss = np.array(list(map(float, list(label_87_loss_to_img.keys())))).reshape(-1, 1)
label_88_loss = np.array(list(map(float, list(label_88_loss_to_img.keys())))).reshape(-1, 1)
label_89_loss = np.array(list(map(float, list(label_89_loss_to_img.keys())))).reshape(-1, 1)


label_90_loss = np.array(list(map(float, list(label_90_loss_to_img.keys())))).reshape(-1, 1)
label_91_loss = np.array(list(map(float, list(label_91_loss_to_img.keys())))).reshape(-1, 1)
label_92_loss = np.array(list(map(float, list(label_92_loss_to_img.keys())))).reshape(-1, 1)
label_93_loss = np.array(list(map(float, list(label_93_loss_to_img.keys())))).reshape(-1, 1)
label_94_loss = np.array(list(map(float, list(label_94_loss_to_img.keys())))).reshape(-1, 1)
label_95_loss = np.array(list(map(float, list(label_95_loss_to_img.keys())))).reshape(-1, 1)
label_96_loss = np.array(list(map(float, list(label_96_loss_to_img.keys())))).reshape(-1, 1)
label_97_loss = np.array(list(map(float, list(label_97_loss_to_img.keys())))).reshape(-1, 1)
label_98_loss = np.array(list(map(float, list(label_98_loss_to_img.keys())))).reshape(-1, 1)
label_99_loss = np.array(list(map(float, list(label_99_loss_to_img.keys())))).reshape(-1, 1)


label_100_loss = np.array(list(map(float, list(label_100_loss_to_img.keys())))).reshape(-1, 1)
label_101_loss = np.array(list(map(float, list(label_101_loss_to_img.keys())))).reshape(-1, 1)
label_102_loss = np.array(list(map(float, list(label_102_loss_to_img.keys())))).reshape(-1, 1)
label_103_loss = np.array(list(map(float, list(label_103_loss_to_img.keys())))).reshape(-1, 1)
label_104_loss = np.array(list(map(float, list(label_104_loss_to_img.keys())))).reshape(-1, 1)
label_105_loss = np.array(list(map(float, list(label_105_loss_to_img.keys())))).reshape(-1, 1)
label_106_loss = np.array(list(map(float, list(label_106_loss_to_img.keys())))).reshape(-1, 1)
label_107_loss = np.array(list(map(float, list(label_107_loss_to_img.keys())))).reshape(-1, 1)
label_108_loss = np.array(list(map(float, list(label_108_loss_to_img.keys())))).reshape(-1, 1)
label_109_loss = np.array(list(map(float, list(label_109_loss_to_img.keys())))).reshape(-1, 1)


label_110_loss = np.array(list(map(float, list(label_110_loss_to_img.keys())))).reshape(-1, 1)
label_111_loss = np.array(list(map(float, list(label_111_loss_to_img.keys())))).reshape(-1, 1)
label_112_loss = np.array(list(map(float, list(label_112_loss_to_img.keys())))).reshape(-1, 1)
label_113_loss = np.array(list(map(float, list(label_113_loss_to_img.keys())))).reshape(-1, 1)
label_114_loss = np.array(list(map(float, list(label_114_loss_to_img.keys())))).reshape(-1, 1)
label_115_loss = np.array(list(map(float, list(label_115_loss_to_img.keys())))).reshape(-1, 1)
label_116_loss = np.array(list(map(float, list(label_116_loss_to_img.keys())))).reshape(-1, 1)
label_117_loss = np.array(list(map(float, list(label_117_loss_to_img.keys())))).reshape(-1, 1)
label_118_loss = np.array(list(map(float, list(label_118_loss_to_img.keys())))).reshape(-1, 1)
label_119_loss = np.array(list(map(float, list(label_119_loss_to_img.keys())))).reshape(-1, 1)


label_120_loss = np.array(list(map(float, list(label_120_loss_to_img.keys())))).reshape(-1, 1)
label_121_loss = np.array(list(map(float, list(label_121_loss_to_img.keys())))).reshape(-1, 1)
label_122_loss = np.array(list(map(float, list(label_122_loss_to_img.keys())))).reshape(-1, 1)
label_123_loss = np.array(list(map(float, list(label_123_loss_to_img.keys())))).reshape(-1, 1)
label_124_loss = np.array(list(map(float, list(label_124_loss_to_img.keys())))).reshape(-1, 1)
label_125_loss = np.array(list(map(float, list(label_125_loss_to_img.keys())))).reshape(-1, 1)
label_126_loss = np.array(list(map(float, list(label_126_loss_to_img.keys())))).reshape(-1, 1)
label_127_loss = np.array(list(map(float, list(label_127_loss_to_img.keys())))).reshape(-1, 1)
label_128_loss = np.array(list(map(float, list(label_128_loss_to_img.keys())))).reshape(-1, 1)
label_129_loss = np.array(list(map(float, list(label_129_loss_to_img.keys())))).reshape(-1, 1)


label_130_loss = np.array(list(map(float, list(label_130_loss_to_img.keys())))).reshape(-1, 1)
label_131_loss = np.array(list(map(float, list(label_131_loss_to_img.keys())))).reshape(-1, 1)
label_132_loss = np.array(list(map(float, list(label_132_loss_to_img.keys())))).reshape(-1, 1)
label_133_loss = np.array(list(map(float, list(label_133_loss_to_img.keys())))).reshape(-1, 1)
label_134_loss = np.array(list(map(float, list(label_134_loss_to_img.keys())))).reshape(-1, 1)
label_135_loss = np.array(list(map(float, list(label_135_loss_to_img.keys())))).reshape(-1, 1)
label_136_loss = np.array(list(map(float, list(label_136_loss_to_img.keys())))).reshape(-1, 1)
label_137_loss = np.array(list(map(float, list(label_137_loss_to_img.keys())))).reshape(-1, 1)
label_138_loss = np.array(list(map(float, list(label_138_loss_to_img.keys())))).reshape(-1, 1)
label_139_loss = np.array(list(map(float, list(label_139_loss_to_img.keys())))).reshape(-1, 1)


label_140_loss = np.array(list(map(float, list(label_140_loss_to_img.keys())))).reshape(-1, 1)
label_141_loss = np.array(list(map(float, list(label_141_loss_to_img.keys())))).reshape(-1, 1)
label_142_loss = np.array(list(map(float, list(label_142_loss_to_img.keys())))).reshape(-1, 1)
label_143_loss = np.array(list(map(float, list(label_143_loss_to_img.keys())))).reshape(-1, 1)
label_144_loss = np.array(list(map(float, list(label_144_loss_to_img.keys())))).reshape(-1, 1)
label_145_loss = np.array(list(map(float, list(label_145_loss_to_img.keys())))).reshape(-1, 1)
label_146_loss = np.array(list(map(float, list(label_146_loss_to_img.keys())))).reshape(-1, 1)
label_147_loss = np.array(list(map(float, list(label_147_loss_to_img.keys())))).reshape(-1, 1)
label_148_loss = np.array(list(map(float, list(label_148_loss_to_img.keys())))).reshape(-1, 1)
label_149_loss = np.array(list(map(float, list(label_149_loss_to_img.keys())))).reshape(-1, 1)


label_150_loss = np.array(list(map(float, list(label_150_loss_to_img.keys())))).reshape(-1, 1)
label_151_loss = np.array(list(map(float, list(label_151_loss_to_img.keys())))).reshape(-1, 1)
label_152_loss = np.array(list(map(float, list(label_152_loss_to_img.keys())))).reshape(-1, 1)
label_153_loss = np.array(list(map(float, list(label_153_loss_to_img.keys())))).reshape(-1, 1)
label_154_loss = np.array(list(map(float, list(label_154_loss_to_img.keys())))).reshape(-1, 1)
label_155_loss = np.array(list(map(float, list(label_155_loss_to_img.keys())))).reshape(-1, 1)
label_156_loss = np.array(list(map(float, list(label_156_loss_to_img.keys())))).reshape(-1, 1)
label_157_loss = np.array(list(map(float, list(label_157_loss_to_img.keys())))).reshape(-1, 1)
label_158_loss = np.array(list(map(float, list(label_158_loss_to_img.keys())))).reshape(-1, 1)
label_159_loss = np.array(list(map(float, list(label_159_loss_to_img.keys())))).reshape(-1, 1)


label_160_loss = np.array(list(map(float, list(label_160_loss_to_img.keys())))).reshape(-1, 1)
label_161_loss = np.array(list(map(float, list(label_161_loss_to_img.keys())))).reshape(-1, 1)
label_162_loss = np.array(list(map(float, list(label_162_loss_to_img.keys())))).reshape(-1, 1)
label_163_loss = np.array(list(map(float, list(label_163_loss_to_img.keys())))).reshape(-1, 1)
label_164_loss = np.array(list(map(float, list(label_164_loss_to_img.keys())))).reshape(-1, 1)
label_165_loss = np.array(list(map(float, list(label_165_loss_to_img.keys())))).reshape(-1, 1)
label_166_loss = np.array(list(map(float, list(label_166_loss_to_img.keys())))).reshape(-1, 1)
label_167_loss = np.array(list(map(float, list(label_167_loss_to_img.keys())))).reshape(-1, 1)
label_168_loss = np.array(list(map(float, list(label_168_loss_to_img.keys())))).reshape(-1, 1)
label_169_loss = np.array(list(map(float, list(label_169_loss_to_img.keys())))).reshape(-1, 1)

label_170_loss = np.array(list(map(float, list(label_170_loss_to_img.keys())))).reshape(-1, 1)
label_171_loss = np.array(list(map(float, list(label_171_loss_to_img.keys())))).reshape(-1, 1)
label_172_loss = np.array(list(map(float, list(label_172_loss_to_img.keys())))).reshape(-1, 1)
label_173_loss = np.array(list(map(float, list(label_173_loss_to_img.keys())))).reshape(-1, 1)
label_174_loss = np.array(list(map(float, list(label_174_loss_to_img.keys())))).reshape(-1, 1)
label_175_loss = np.array(list(map(float, list(label_175_loss_to_img.keys())))).reshape(-1, 1)
label_176_loss = np.array(list(map(float, list(label_176_loss_to_img.keys())))).reshape(-1, 1)
label_177_loss = np.array(list(map(float, list(label_177_loss_to_img.keys())))).reshape(-1, 1)
label_178_loss = np.array(list(map(float, list(label_178_loss_to_img.keys())))).reshape(-1, 1)
label_179_loss = np.array(list(map(float, list(label_179_loss_to_img.keys())))).reshape(-1, 1)


label_180_loss = np.array(list(map(float, list(label_180_loss_to_img.keys())))).reshape(-1, 1)
label_181_loss = np.array(list(map(float, list(label_181_loss_to_img.keys())))).reshape(-1, 1)
label_182_loss = np.array(list(map(float, list(label_182_loss_to_img.keys())))).reshape(-1, 1)
label_183_loss = np.array(list(map(float, list(label_183_loss_to_img.keys())))).reshape(-1, 1)
label_184_loss = np.array(list(map(float, list(label_184_loss_to_img.keys())))).reshape(-1, 1)
label_185_loss = np.array(list(map(float, list(label_185_loss_to_img.keys())))).reshape(-1, 1)
label_186_loss = np.array(list(map(float, list(label_186_loss_to_img.keys())))).reshape(-1, 1)
label_187_loss = np.array(list(map(float, list(label_187_loss_to_img.keys())))).reshape(-1, 1)
label_188_loss = np.array(list(map(float, list(label_188_loss_to_img.keys())))).reshape(-1, 1)
label_189_loss = np.array(list(map(float, list(label_189_loss_to_img.keys())))).reshape(-1, 1)



label_190_loss = np.array(list(map(float, list(label_190_loss_to_img.keys())))).reshape(-1, 1)
label_191_loss = np.array(list(map(float, list(label_191_loss_to_img.keys())))).reshape(-1, 1)
label_192_loss = np.array(list(map(float, list(label_192_loss_to_img.keys())))).reshape(-1, 1)
label_193_loss = np.array(list(map(float, list(label_193_loss_to_img.keys())))).reshape(-1, 1)
label_194_loss = np.array(list(map(float, list(label_194_loss_to_img.keys())))).reshape(-1, 1)
label_195_loss = np.array(list(map(float, list(label_195_loss_to_img.keys())))).reshape(-1, 1)
label_196_loss = np.array(list(map(float, list(label_196_loss_to_img.keys())))).reshape(-1, 1)
label_197_loss = np.array(list(map(float, list(label_197_loss_to_img.keys())))).reshape(-1, 1)
label_198_loss = np.array(list(map(float, list(label_198_loss_to_img.keys())))).reshape(-1, 1)
label_199_loss = np.array(list(map(float, list(label_199_loss_to_img.keys())))).reshape(-1, 1)


label_200_loss = np.array(list(map(float, list(label_200_loss_to_img.keys())))).reshape(-1, 1)
label_201_loss = np.array(list(map(float, list(label_201_loss_to_img.keys())))).reshape(-1, 1)
label_202_loss = np.array(list(map(float, list(label_202_loss_to_img.keys())))).reshape(-1, 1)
label_203_loss = np.array(list(map(float, list(label_203_loss_to_img.keys())))).reshape(-1, 1)
label_204_loss = np.array(list(map(float, list(label_204_loss_to_img.keys())))).reshape(-1, 1)
label_205_loss = np.array(list(map(float, list(label_205_loss_to_img.keys())))).reshape(-1, 1)
label_206_loss = np.array(list(map(float, list(label_206_loss_to_img.keys())))).reshape(-1, 1)
label_207_loss = np.array(list(map(float, list(label_207_loss_to_img.keys())))).reshape(-1, 1)
label_208_loss = np.array(list(map(float, list(label_208_loss_to_img.keys())))).reshape(-1, 1)
label_209_loss = np.array(list(map(float, list(label_209_loss_to_img.keys())))).reshape(-1, 1)


label_210_loss = np.array(list(map(float, list(label_210_loss_to_img.keys())))).reshape(-1, 1)
label_211_loss = np.array(list(map(float, list(label_211_loss_to_img.keys())))).reshape(-1, 1)
label_212_loss = np.array(list(map(float, list(label_212_loss_to_img.keys())))).reshape(-1, 1)
label_213_loss = np.array(list(map(float, list(label_213_loss_to_img.keys())))).reshape(-1, 1)
label_214_loss = np.array(list(map(float, list(label_214_loss_to_img.keys())))).reshape(-1, 1)
label_215_loss = np.array(list(map(float, list(label_215_loss_to_img.keys())))).reshape(-1, 1)
label_216_loss = np.array(list(map(float, list(label_216_loss_to_img.keys())))).reshape(-1, 1)
label_217_loss = np.array(list(map(float, list(label_217_loss_to_img.keys())))).reshape(-1, 1)
label_218_loss = np.array(list(map(float, list(label_218_loss_to_img.keys())))).reshape(-1, 1)
label_219_loss = np.array(list(map(float, list(label_219_loss_to_img.keys())))).reshape(-1, 1)


label_220_loss = np.array(list(map(float, list(label_220_loss_to_img.keys())))).reshape(-1, 1)
label_221_loss = np.array(list(map(float, list(label_221_loss_to_img.keys())))).reshape(-1, 1)
label_222_loss = np.array(list(map(float, list(label_222_loss_to_img.keys())))).reshape(-1, 1)
label_223_loss = np.array(list(map(float, list(label_223_loss_to_img.keys())))).reshape(-1, 1)
label_224_loss = np.array(list(map(float, list(label_224_loss_to_img.keys())))).reshape(-1, 1)
label_225_loss = np.array(list(map(float, list(label_225_loss_to_img.keys())))).reshape(-1, 1)
label_226_loss = np.array(list(map(float, list(label_226_loss_to_img.keys())))).reshape(-1, 1)
label_227_loss = np.array(list(map(float, list(label_227_loss_to_img.keys())))).reshape(-1, 1)
label_228_loss = np.array(list(map(float, list(label_228_loss_to_img.keys())))).reshape(-1, 1)
label_229_loss = np.array(list(map(float, list(label_229_loss_to_img.keys())))).reshape(-1, 1)


label_230_loss = np.array(list(map(float, list(label_230_loss_to_img.keys())))).reshape(-1, 1)
label_231_loss = np.array(list(map(float, list(label_231_loss_to_img.keys())))).reshape(-1, 1)
label_232_loss = np.array(list(map(float, list(label_232_loss_to_img.keys())))).reshape(-1, 1)
label_233_loss = np.array(list(map(float, list(label_233_loss_to_img.keys())))).reshape(-1, 1)
label_234_loss = np.array(list(map(float, list(label_234_loss_to_img.keys())))).reshape(-1, 1)
label_235_loss = np.array(list(map(float, list(label_235_loss_to_img.keys())))).reshape(-1, 1)
label_236_loss = np.array(list(map(float, list(label_236_loss_to_img.keys())))).reshape(-1, 1)
label_237_loss = np.array(list(map(float, list(label_237_loss_to_img.keys())))).reshape(-1, 1)
label_238_loss = np.array(list(map(float, list(label_238_loss_to_img.keys())))).reshape(-1, 1)
label_239_loss = np.array(list(map(float, list(label_239_loss_to_img.keys())))).reshape(-1, 1)

label_240_loss = np.array(list(map(float, list(label_240_loss_to_img.keys())))).reshape(-1, 1)
label_241_loss = np.array(list(map(float, list(label_241_loss_to_img.keys())))).reshape(-1, 1)
label_242_loss = np.array(list(map(float, list(label_242_loss_to_img.keys())))).reshape(-1, 1)
label_243_loss = np.array(list(map(float, list(label_243_loss_to_img.keys())))).reshape(-1, 1)
label_244_loss = np.array(list(map(float, list(label_244_loss_to_img.keys())))).reshape(-1, 1)
label_245_loss = np.array(list(map(float, list(label_245_loss_to_img.keys())))).reshape(-1, 1)
label_246_loss = np.array(list(map(float, list(label_246_loss_to_img.keys())))).reshape(-1, 1)
label_247_loss = np.array(list(map(float, list(label_247_loss_to_img.keys())))).reshape(-1, 1)
label_248_loss = np.array(list(map(float, list(label_248_loss_to_img.keys())))).reshape(-1, 1)
label_249_loss = np.array(list(map(float, list(label_249_loss_to_img.keys())))).reshape(-1, 1)

label_250_loss = np.array(list(map(float, list(label_250_loss_to_img.keys())))).reshape(-1, 1)
label_251_loss = np.array(list(map(float, list(label_251_loss_to_img.keys())))).reshape(-1, 1)
label_252_loss = np.array(list(map(float, list(label_252_loss_to_img.keys())))).reshape(-1, 1)
label_253_loss = np.array(list(map(float, list(label_253_loss_to_img.keys())))).reshape(-1, 1)
label_254_loss = np.array(list(map(float, list(label_254_loss_to_img.keys())))).reshape(-1, 1)
label_255_loss = np.array(list(map(float, list(label_255_loss_to_img.keys())))).reshape(-1, 1)
label_256_loss = np.array(list(map(float, list(label_256_loss_to_img.keys())))).reshape(-1, 1)


imgs = []
# ==================================================针对第0类的采样=================================================

selected_sampled_losses_0 = []

ms_0 = MeanShift()
ms_0.fit(label_0_loss)

labels_0 = ms_0.labels_  # 第0类中的每个样本对应的簇号

n_clusters_0 = len(np.unique(labels_0))  # 第0类样本有几个簇

need_sampled_cluster_0 = []  # 需要下采样的簇号

cluster_to_num_0 = Counter(labels_0)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_0.keys():
    if cluster_to_num_0[k] > len(labels_0) // n_clusters_0:
        need_sampled_cluster_0.append(k)

need_sampled_losses_0 = [[] for _ in range(len(need_sampled_cluster_0))]

for i in range(len(need_sampled_cluster_0)):
    for j in range(len(labels_0)):
        if labels_0[j] == need_sampled_cluster_0[i]:
            need_sampled_losses_0[i].append(label_0_loss[j][0])

for j in range(len(labels_0)):
    if labels_0[j] not in need_sampled_cluster_0:
        selected_sampled_losses_0.append(label_0_loss[j][0])

for loss in need_sampled_losses_0:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_0.append(lo)
print(len(label_0_loss))
print(len(set(selected_sampled_losses_0)))

for loss in list(set(selected_sampled_losses_0)):
    for img in label_0_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第1类的采样=================================================
selected_sampled_losses_1 = []

ms_1 = MeanShift()
ms_1.fit(label_1_loss)

labels_1 = ms_1.labels_  # 第0类中的每个样本对应的簇号

n_clusters_1 = len(np.unique(labels_1))  # 第0类样本有几个簇

need_sampled_cluster_1 = []  # 需要下采样的簇号

cluster_to_num_1 = Counter(labels_1)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_1.keys():
    if cluster_to_num_1[k] > len(labels_1) // n_clusters_1:
        need_sampled_cluster_1.append(k)

need_sampled_losses_1 = [[] for _ in range(len(need_sampled_cluster_1))]

for i in range(len(need_sampled_cluster_1)):
    for j in range(len(labels_1)):
        if labels_1[j] == need_sampled_cluster_1[i]:
            need_sampled_losses_1[i].append(label_1_loss[j][0])

for j in range(len(labels_1)):
    if labels_1[j] not in need_sampled_cluster_1:
        selected_sampled_losses_1.append(label_1_loss[j][0])

for loss in need_sampled_losses_1:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_1.append(lo)
print(len(label_1_loss))
print(len(set(selected_sampled_losses_1)))

for loss in list(set(selected_sampled_losses_1)):
    for img in label_1_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第2类的采样=================================================
selected_sampled_losses_2 = []

ms_2 = MeanShift()
ms_2.fit(label_2_loss)

labels_2 = ms_2.labels_  # 第0类中的每个样本对应的簇号

n_clusters_2 = len(np.unique(labels_2))  # 第0类样本有几个簇

need_sampled_cluster_2 = []  # 需要下采样的簇号

cluster_to_num_2 = Counter(labels_2)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_2.keys():
    if cluster_to_num_2[k] > len(labels_2) // n_clusters_2:
        need_sampled_cluster_2.append(k)

need_sampled_losses_2 = [[] for _ in range(len(need_sampled_cluster_2))]

for i in range(len(need_sampled_cluster_2)):
    for j in range(len(labels_2)):
        if labels_2[j] == need_sampled_cluster_2[i]:
            need_sampled_losses_2[i].append(label_2_loss[j][0])

for j in range(len(labels_2)):
    if labels_2[j] not in need_sampled_cluster_2:
        selected_sampled_losses_2.append(label_2_loss[j][0])

for loss in need_sampled_losses_2:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_2.append(lo)
print(len(label_2_loss))
print(len(set(selected_sampled_losses_2)))

for loss in list(set(selected_sampled_losses_2)):
    for img in label_2_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第3类的采样=================================================
selected_sampled_losses_3 = []

ms_3 = MeanShift()
ms_3.fit(label_3_loss)

labels_3 = ms_3.labels_  # 第0类中的每个样本对应的簇号

n_clusters_3 = len(np.unique(labels_3))  # 第0类样本有几个簇

need_sampled_cluster_3 = []  # 需要下采样的簇号

cluster_to_num_3 = Counter(labels_3)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_3.keys():
    if cluster_to_num_3[k] > len(labels_3) // n_clusters_3:
        need_sampled_cluster_3.append(k)

need_sampled_losses_3 = [[] for _ in range(len(need_sampled_cluster_3))]

for i in range(len(need_sampled_cluster_3)):
    for j in range(len(labels_3)):
        if labels_3[j] == need_sampled_cluster_3[i]:
            need_sampled_losses_3[i].append(label_3_loss[j][0])

for j in range(len(labels_3)):
    if labels_3[j] not in need_sampled_cluster_3:
        selected_sampled_losses_3.append(label_3_loss[j][0])

for loss in need_sampled_losses_3:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_3.append(lo)
print(len(label_3_loss))
print(len(set(selected_sampled_losses_3)))

for loss in list(set(selected_sampled_losses_3)):
    for img in label_3_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第4类的采样=================================================
selected_sampled_losses_4 = []
ms_4 = MeanShift()
ms_4.fit(label_4_loss)

labels_4 = ms_4.labels_  # 第0类中的每个样本对应的簇号

n_clusters_4 = len(np.unique(labels_4))  # 第0类样本有几个簇

need_sampled_cluster_4 = []  # 需要下采样的簇号

cluster_to_num_4 = Counter(labels_4)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_4.keys():
    if cluster_to_num_4[k] > len(labels_4) // n_clusters_4:
        need_sampled_cluster_4.append(k)

need_sampled_losses_4 = [[] for _ in range(len(need_sampled_cluster_4))]

for i in range(len(need_sampled_cluster_4)):
    for j in range(len(labels_4)):
        if labels_4[j] == need_sampled_cluster_4[i]:
            need_sampled_losses_4[i].append(label_4_loss[j][0])

for j in range(len(labels_4)):
    if labels_4[j] not in need_sampled_cluster_4:
        selected_sampled_losses_4.append(label_4_loss[j][0])

for loss in need_sampled_losses_4:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_4.append(lo)
print(len(label_4_loss))
print(len(set(selected_sampled_losses_4)))

for loss in list(set(selected_sampled_losses_4)):
    for img in label_4_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第5类的采样=================================================
selected_sampled_losses_5 = []
ms_5 = MeanShift()
ms_5.fit(label_5_loss)

labels_5 = ms_5.labels_  # 第0类中的每个样本对应的簇号

n_clusters_5 = len(np.unique(labels_5))  # 第0类样本有几个簇

need_sampled_cluster_5 = []  # 需要下采样的簇号

cluster_to_num_5 = Counter(labels_5)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_5.keys():
    if cluster_to_num_5[k] > len(labels_5) // n_clusters_5:
        need_sampled_cluster_5.append(k)

need_sampled_losses_5 = [[] for _ in range(len(need_sampled_cluster_5))]

for i in range(len(need_sampled_cluster_5)):
    for j in range(len(labels_5)):
        if labels_5[j] == need_sampled_cluster_5[i]:
            need_sampled_losses_5[i].append(label_5_loss[j][0])

for j in range(len(labels_5)):
    if labels_5[j] not in need_sampled_cluster_5:
        selected_sampled_losses_5.append(label_5_loss[j][0])

for loss in need_sampled_losses_5:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_5.append(lo)
print(len(label_5_loss))
print(len(set(selected_sampled_losses_5)))

for loss in list(set(selected_sampled_losses_5)):
    for img in label_5_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第6类的采样=================================================
selected_sampled_losses_6 = []
ms_6 = MeanShift()
ms_6.fit(label_6_loss)

labels_6 = ms_6.labels_  # 第0类中的每个样本对应的簇号

n_clusters_6 = len(np.unique(labels_6))  # 第0类样本有几个簇

need_sampled_cluster_6 = []  # 需要下采样的簇号

cluster_to_num_6 = Counter(labels_6)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_6.keys():
    if cluster_to_num_6[k] > len(labels_6) // n_clusters_6:
        need_sampled_cluster_6.append(k)

need_sampled_losses_6 = [[] for _ in range(len(need_sampled_cluster_6))]

for i in range(len(need_sampled_cluster_6)):
    for j in range(len(labels_6)):
        if labels_6[j] == need_sampled_cluster_6[i]:
            need_sampled_losses_6[i].append(label_6_loss[j][0])

for j in range(len(labels_6)):
    if labels_6[j] not in need_sampled_cluster_6:
        selected_sampled_losses_6.append(label_6_loss[j][0])

for loss in need_sampled_losses_6:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_6.append(lo)
print(len(label_6_loss))
print(len(set(selected_sampled_losses_6)))

for loss in list(set(selected_sampled_losses_6)):
    for img in label_6_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第7类的采样=================================================
selected_sampled_losses_7 = []
ms_7 = MeanShift()
ms_7.fit(label_7_loss)

labels_7 = ms_7.labels_  # 第0类中的每个样本对应的簇号

n_clusters_7 = len(np.unique(labels_7))  # 第0类样本有几个簇

need_sampled_cluster_7 = []  # 需要下采样的簇号

cluster_to_num_7 = Counter(labels_7)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_7.keys():
    if cluster_to_num_7[k] > len(labels_7) // n_clusters_7:
        need_sampled_cluster_7.append(k)

need_sampled_losses_7 = [[] for _ in range(len(need_sampled_cluster_7))]

for i in range(len(need_sampled_cluster_7)):
    for j in range(len(labels_7)):
        if labels_7[j] == need_sampled_cluster_7[i]:
            need_sampled_losses_7[i].append(label_7_loss[j][0])

for j in range(len(labels_7)):
    if labels_7[j] not in need_sampled_cluster_7:
        selected_sampled_losses_7.append(label_7_loss[j][0])

for loss in need_sampled_losses_7:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_7.append(lo)
print(len(label_7_loss))
print(len(set(selected_sampled_losses_7)))

for loss in list(set(selected_sampled_losses_7)):
    for img in label_7_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第8类的采样=================================================
selected_sampled_losses_8 = []
ms_8 = MeanShift()
ms_8.fit(label_8_loss)

labels_8 = ms_8.labels_  # 第0类中的每个样本对应的簇号

n_clusters_8 = len(np.unique(labels_8))  # 第0类样本有几个簇

need_sampled_cluster_8 = []  # 需要下采样的簇号

cluster_to_num_8 = Counter(labels_8)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_8.keys():
    if cluster_to_num_8[k] > len(labels_8) // n_clusters_8:
        need_sampled_cluster_8.append(k)

need_sampled_losses_8 = [[] for _ in range(len(need_sampled_cluster_8))]

for i in range(len(need_sampled_cluster_8)):
    for j in range(len(labels_8)):
        if labels_8[j] == need_sampled_cluster_8[i]:
            need_sampled_losses_8[i].append(label_8_loss[j][0])

for j in range(len(labels_8)):
    if labels_8[j] not in need_sampled_cluster_8:
        selected_sampled_losses_8.append(label_8_loss[j][0])

for loss in need_sampled_losses_8:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_8.append(lo)
print(len(label_8_loss))
print(len(set(selected_sampled_losses_8)))

for loss in list(set(selected_sampled_losses_8)):
    for img in label_8_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第9类的采样=================================================
selected_sampled_losses_9 = []
ms_9 = MeanShift()
ms_9.fit(label_9_loss)

labels_9 = ms_9.labels_  # 第0类中的每个样本对应的簇号

n_clusters_9 = len(np.unique(labels_9))  # 第0类样本有几个簇

need_sampled_cluster_9 = []  # 需要下采样的簇号

cluster_to_num_9 = Counter(labels_9)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_9.keys():
    if cluster_to_num_9[k] > len(labels_9) // n_clusters_9:
        need_sampled_cluster_9.append(k)

need_sampled_losses_9 = [[] for _ in range(len(need_sampled_cluster_9))]

for i in range(len(need_sampled_cluster_9)):
    for j in range(len(labels_9)):
        if labels_9[j] == need_sampled_cluster_9[i]:
            need_sampled_losses_9[i].append(label_9_loss[j][0])

for j in range(len(labels_9)):
    if labels_9[j] not in need_sampled_cluster_9:
        selected_sampled_losses_9.append(label_9_loss[j][0])

for loss in need_sampled_losses_9:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_9.append(lo)
print(len(label_9_loss))
print(len(set(selected_sampled_losses_9)))

for loss in list(set(selected_sampled_losses_9)):
    for img in label_9_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第10类的采样=================================================
selected_sampled_losses_10 = []
ms_10 = MeanShift()
ms_10.fit(label_10_loss)

labels_10 = ms_10.labels_  # 第0类中的每个样本对应的簇号

n_clusters_10 = len(np.unique(labels_10))  # 第0类样本有几个簇

need_sampled_cluster_10 = []  # 需要下采样的簇号

cluster_to_num_10 = Counter(labels_10)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_10.keys():
    if cluster_to_num_10[k] > len(labels_10) // n_clusters_10:
        need_sampled_cluster_10.append(k)

need_sampled_losses_10 = [[] for _ in range(len(need_sampled_cluster_10))]

for i in range(len(need_sampled_cluster_10)):
    for j in range(len(labels_10)):
        if labels_10[j] == need_sampled_cluster_10[i]:
            need_sampled_losses_10[i].append(label_10_loss[j][0])

for j in range(len(labels_10)):
    if labels_10[j] not in need_sampled_cluster_10:
        selected_sampled_losses_10.append(label_10_loss[j][0])

for loss in need_sampled_losses_10:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_10.append(lo)
print(len(label_10_loss))
print(len(set(selected_sampled_losses_10)))

for loss in list(set(selected_sampled_losses_10)):
    for img in label_10_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第11类的采样=================================================
selected_sampled_losses_11 = []

ms_11 = MeanShift()
ms_11.fit(label_11_loss)

labels_11 = ms_11.labels_  # 第0类中的每个样本对应的簇号

n_clusters_11 = len(np.unique(labels_11))  # 第0类样本有几个簇

need_sampled_cluster_11 = []  # 需要下采样的簇号

cluster_to_num_11 = Counter(labels_11)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_11.keys():
    if cluster_to_num_11[k] > len(labels_11) // n_clusters_11:
        need_sampled_cluster_11.append(k)

need_sampled_losses_11 = [[] for _ in range(len(need_sampled_cluster_11))]

for i in range(len(need_sampled_cluster_11)):
    for j in range(len(labels_11)):
        if labels_11[j] == need_sampled_cluster_11[i]:
            need_sampled_losses_11[i].append(label_11_loss[j][0])

for j in range(len(labels_11)):
    if labels_11[j] not in need_sampled_cluster_11:
        selected_sampled_losses_11.append(label_11_loss[j][0])

for loss in need_sampled_losses_11:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_11.append(lo)
print(len(label_11_loss))
print(len(set(selected_sampled_losses_11)))

for loss in list(set(selected_sampled_losses_11)):
    for img in label_11_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第12类的采样=================================================
selected_sampled_losses_12 = []
ms_12 = MeanShift()
ms_12.fit(label_12_loss)

labels_12 = ms_12.labels_  # 第0类中的每个样本对应的簇号

n_clusters_12 = len(np.unique(labels_12))  # 第0类样本有几个簇

need_sampled_cluster_12 = []  # 需要下采样的簇号

cluster_to_num_12 = Counter(labels_12)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_12.keys():
    if cluster_to_num_12[k] > len(labels_12) // n_clusters_12:
        need_sampled_cluster_12.append(k)

need_sampled_losses_12 = [[] for _ in range(len(need_sampled_cluster_12))]

for i in range(len(need_sampled_cluster_12)):
    for j in range(len(labels_12)):
        if labels_12[j] == need_sampled_cluster_12[i]:
            need_sampled_losses_12[i].append(label_12_loss[j][0])

for j in range(len(labels_12)):
    if labels_12[j] not in need_sampled_cluster_12:
        selected_sampled_losses_12.append(label_12_loss[j][0])

for loss in need_sampled_losses_12:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_12.append(lo)
print(len(label_12_loss))
print(len(set(selected_sampled_losses_12)))

for loss in list(set(selected_sampled_losses_12)):
    for img in label_12_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第13类的采样=================================================
selected_sampled_losses_13 = []
ms_13 = MeanShift()
ms_13.fit(label_13_loss)

labels_13 = ms_13.labels_  # 第0类中的每个样本对应的簇号

n_clusters_13 = len(np.unique(labels_13))  # 第0类样本有几个簇

need_sampled_cluster_13 = []  # 需要下采样的簇号

cluster_to_num_13 = Counter(labels_13)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_13.keys():
    if cluster_to_num_13[k] > len(labels_13) // n_clusters_13:
        need_sampled_cluster_13.append(k)

need_sampled_losses_13 = [[] for _ in range(len(need_sampled_cluster_13))]

for i in range(len(need_sampled_cluster_13)):
    for j in range(len(labels_13)):
        if labels_13[j] == need_sampled_cluster_13[i]:
            need_sampled_losses_13[i].append(label_13_loss[j][0])

for j in range(len(labels_13)):
    if labels_13[j] not in need_sampled_cluster_13:
        selected_sampled_losses_13.append(label_13_loss[j][0])

for loss in need_sampled_losses_13:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_13.append(lo)
print(len(label_13_loss))
print(len(set(selected_sampled_losses_13)))

for loss in list(set(selected_sampled_losses_13)):
    for img in label_13_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第14类的采样=================================================
selected_sampled_losses_14 = []
ms_14 = MeanShift()
ms_14.fit(label_14_loss)

labels_14 = ms_14.labels_  # 第0类中的每个样本对应的簇号

n_clusters_14 = len(np.unique(labels_14))  # 第0类样本有几个簇

need_sampled_cluster_14 = []  # 需要下采样的簇号

cluster_to_num_14 = Counter(labels_14)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_14.keys():
    if cluster_to_num_14[k] > len(labels_14) // n_clusters_14:
        need_sampled_cluster_14.append(k)

need_sampled_losses_14 = [[] for _ in range(len(need_sampled_cluster_14))]

for i in range(len(need_sampled_cluster_14)):
    for j in range(len(labels_14)):
        if labels_14[j] == need_sampled_cluster_14[i]:
            need_sampled_losses_14[i].append(label_14_loss[j][0])

for j in range(len(labels_14)):
    if labels_14[j] not in need_sampled_cluster_14:
        selected_sampled_losses_14.append(label_14_loss[j][0])

for loss in need_sampled_losses_14:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_14.append(lo)
print(len(label_14_loss))
print(len(set(selected_sampled_losses_14)))

for loss in list(set(selected_sampled_losses_14)):
    for img in label_14_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第15类的采样=================================================
selected_sampled_losses_15 = []
ms_15 = MeanShift()
ms_15.fit(label_15_loss)

labels_15 = ms_15.labels_  # 第0类中的每个样本对应的簇号

n_clusters_15 = len(np.unique(labels_15))  # 第0类样本有几个簇

need_sampled_cluster_15 = []  # 需要下采样的簇号

cluster_to_num_15 = Counter(labels_15)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_15.keys():
    if cluster_to_num_15[k] > len(labels_15) // n_clusters_15:
        need_sampled_cluster_15.append(k)

need_sampled_losses_15 = [[] for _ in range(len(need_sampled_cluster_15))]

for i in range(len(need_sampled_cluster_15)):
    for j in range(len(labels_15)):
        if labels_15[j] == need_sampled_cluster_15[i]:
            need_sampled_losses_15[i].append(label_15_loss[j][0])

for j in range(len(labels_15)):
    if labels_15[j] not in need_sampled_cluster_15:
        selected_sampled_losses_15.append(label_15_loss[j][0])

for loss in need_sampled_losses_15:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_15.append(lo)
print(len(label_15_loss))
print(len(set(selected_sampled_losses_15)))

for loss in list(set(selected_sampled_losses_15)):
    for img in label_15_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第16类的采样=================================================
selected_sampled_losses_16 = []
ms_16 = MeanShift()
ms_16.fit(label_16_loss)

labels_16 = ms_16.labels_  # 第0类中的每个样本对应的簇号

n_clusters_16 = len(np.unique(labels_16))  # 第0类样本有几个簇

need_sampled_cluster_16 = []  # 需要下采样的簇号

cluster_to_num_16 = Counter(labels_16)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_16.keys():
    if cluster_to_num_16[k] > len(labels_16) // n_clusters_16:
        need_sampled_cluster_16.append(k)

need_sampled_losses_16 = [[] for _ in range(len(need_sampled_cluster_16))]

for i in range(len(need_sampled_cluster_16)):
    for j in range(len(labels_16)):
        if labels_16[j] == need_sampled_cluster_16[i]:
            need_sampled_losses_16[i].append(label_16_loss[j][0])

for j in range(len(labels_16)):
    if labels_16[j] not in need_sampled_cluster_16:
        selected_sampled_losses_16.append(label_16_loss[j][0])

for loss in need_sampled_losses_16:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_16.append(lo)
print(len(label_16_loss))
print(len(set(selected_sampled_losses_16)))

for loss in list(set(selected_sampled_losses_16)):
    for img in label_16_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第17类的采样=================================================
selected_sampled_losses_17 = []
ms_17 = MeanShift()
ms_17.fit(label_17_loss)

labels_17 = ms_17.labels_  # 第0类中的每个样本对应的簇号

n_clusters_17 = len(np.unique(labels_17))  # 第0类样本有几个簇

need_sampled_cluster_17 = []  # 需要下采样的簇号

cluster_to_num_17 = Counter(labels_17)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_17.keys():
    if cluster_to_num_17[k] > len(labels_17) // n_clusters_17:
        need_sampled_cluster_17.append(k)

need_sampled_losses_17 = [[] for _ in range(len(need_sampled_cluster_17))]

for i in range(len(need_sampled_cluster_17)):
    for j in range(len(labels_17)):
        if labels_17[j] == need_sampled_cluster_17[i]:
            need_sampled_losses_17[i].append(label_17_loss[j][0])

for j in range(len(labels_17)):
    if labels_17[j] not in need_sampled_cluster_17:
        selected_sampled_losses_17.append(label_17_loss[j][0])

for loss in need_sampled_losses_17:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_17.append(lo)
print(len(label_17_loss))
print(len(set(selected_sampled_losses_17)))

for loss in list(set(selected_sampled_losses_17)):
    for img in label_17_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第18类的采样=================================================
selected_sampled_losses_18 = []
ms_18 = MeanShift()
ms_18.fit(label_18_loss)

labels_18 = ms_18.labels_  # 第0类中的每个样本对应的簇号

n_clusters_18 = len(np.unique(labels_18))  # 第0类样本有几个簇

need_sampled_cluster_18 = []  # 需要下采样的簇号

cluster_to_num_18 = Counter(labels_18)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_18.keys():
    if cluster_to_num_18[k] > len(labels_18) // n_clusters_18:
        need_sampled_cluster_18.append(k)

need_sampled_losses_18 = [[] for _ in range(len(need_sampled_cluster_18))]

for i in range(len(need_sampled_cluster_18)):
    for j in range(len(labels_18)):
        if labels_18[j] == need_sampled_cluster_18[i]:
            need_sampled_losses_18[i].append(label_18_loss[j][0])

for j in range(len(labels_18)):
    if labels_18[j] not in need_sampled_cluster_18:
        selected_sampled_losses_18.append(label_18_loss[j][0])

for loss in need_sampled_losses_18:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_18.append(lo)
print(len(label_18_loss))
print(len(set(selected_sampled_losses_18)))

for loss in list(set(selected_sampled_losses_18)):
    for img in label_18_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第19类的采样=================================================
selected_sampled_losses_19 = []
ms_19 = MeanShift()
ms_19.fit(label_19_loss)

labels_19 = ms_19.labels_  # 第0类中的每个样本对应的簇号

n_clusters_19 = len(np.unique(labels_19))  # 第0类样本有几个簇

need_sampled_cluster_19 = []  # 需要下采样的簇号

cluster_to_num_19 = Counter(labels_19)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_19.keys():
    if cluster_to_num_19[k] > len(labels_19) // n_clusters_19:
        need_sampled_cluster_19.append(k)

need_sampled_losses_19 = [[] for _ in range(len(need_sampled_cluster_19))]

for i in range(len(need_sampled_cluster_19)):
    for j in range(len(labels_19)):
        if labels_19[j] == need_sampled_cluster_19[i]:
            need_sampled_losses_19[i].append(label_19_loss[j][0])

for j in range(len(labels_19)):
    if labels_19[j] not in need_sampled_cluster_19:
        selected_sampled_losses_19.append(label_19_loss[j][0])

for loss in need_sampled_losses_19:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_19.append(lo)
print(len(label_19_loss))
print(len(set(selected_sampled_losses_19)))
for loss in list(set(selected_sampled_losses_19)):
    for img in label_19_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))
        
# ==================================================针对第20类的采样=================================================

selected_sampled_losses_20 = []

ms_20 = MeanShift()
ms_20.fit(label_20_loss)

labels_20 = ms_20.labels_  # 第_20类中的每个样本对应的簇号

n_clusters_20 = len(np.unique(labels_20))  # 第_20类样本有几个簇

need_sampled_cluster_20 = []  # 需要下采样的簇号

cluster_to_num_20 = Counter(labels_20)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_20.keys():
    if cluster_to_num_20[k] > len(labels_20) // n_clusters_20:
        need_sampled_cluster_20.append(k)

need_sampled_losses_20 = [[] for _ in range(len(need_sampled_cluster_20))]

for i in range(len(need_sampled_cluster_20)):
    for j in range(len(labels_20)):
        if labels_20[j] == need_sampled_cluster_20[i]:
            need_sampled_losses_20[i].append(label_20_loss[j][0])

for j in range(len(labels_20)):
    if labels_20[j] not in need_sampled_cluster_20:
        selected_sampled_losses_20.append(label_20_loss[j][0])

for loss in need_sampled_losses_20:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_20.append(lo)
print(len(label_20_loss))
print(len(set(selected_sampled_losses_20)))

for loss in list(set(selected_sampled_losses_20)):
    for img in label_20_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))


# ==================================================针对第21类的采样=================================================
selected_sampled_losses_21 = []

ms_21 = MeanShift()
ms_21.fit(label_21_loss)

labels_21 = ms_21.labels_  # 第0类中的每个样本对应的簇号

n_clusters_21 = len(np.unique(labels_21))  # 第0类样本有几个簇

need_sampled_cluster_21 = []  # 需要下采样的簇号

cluster_to_num_21 = Counter(labels_21)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_21.keys():
    if cluster_to_num_21[k] > len(labels_21) // n_clusters_21:
        need_sampled_cluster_21.append(k)

need_sampled_losses_21 = [[] for _ in range(len(need_sampled_cluster_21))]

for i in range(len(need_sampled_cluster_21)):
    for j in range(len(labels_21)):
        if labels_21[j] == need_sampled_cluster_21[i]:
            need_sampled_losses_21[i].append(label_21_loss[j][0])

for j in range(len(labels_21)):
    if labels_21[j] not in need_sampled_cluster_21:
        selected_sampled_losses_21.append(label_21_loss[j][0])

for loss in need_sampled_losses_21:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_21.append(lo)
print(len(label_21_loss))
print(len(set(selected_sampled_losses_21)))

for loss in list(set(selected_sampled_losses_21)):
    for img in label_21_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第22类的采样=================================================
selected_sampled_losses_22 = []

ms_22 = MeanShift()
ms_22.fit(label_22_loss)

labels_22 = ms_22.labels_  # 第0类中的每个样本对应的簇号

n_clusters_22 = len(np.unique(labels_22))  # 第0类样本有几个簇

need_sampled_cluster_22 = []  # 需要下采样的簇号

cluster_to_num_22 = Counter(labels_22)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_22.keys():
    if cluster_to_num_22[k] > len(labels_22) // n_clusters_22:
        need_sampled_cluster_22.append(k)

need_sampled_losses_22 = [[] for _ in range(len(need_sampled_cluster_22))]

for i in range(len(need_sampled_cluster_22)):
    for j in range(len(labels_22)):
        if labels_22[j] == need_sampled_cluster_22[i]:
            need_sampled_losses_22[i].append(label_22_loss[j][0])

for j in range(len(labels_22)):
    if labels_22[j] not in need_sampled_cluster_22:
        selected_sampled_losses_22.append(label_22_loss[j][0])

for loss in need_sampled_losses_22:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_22.append(lo)
print(len(label_22_loss))
print(len(set(selected_sampled_losses_22)))

for loss in list(set(selected_sampled_losses_22)):
    for img in label_22_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第23类的采样=================================================
selected_sampled_losses_23 = []

ms_23 = MeanShift()
ms_23.fit(label_23_loss)

labels_23 = ms_23.labels_  # 第0类中的每个样本对应的簇号

n_clusters_23 = len(np.unique(labels_23))  # 第0类样本有几个簇

need_sampled_cluster_23 = []  # 需要下采样的簇号

cluster_to_num_23 = Counter(labels_23)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_23.keys():
    if cluster_to_num_23[k] > len(labels_23) // n_clusters_23:
        need_sampled_cluster_23.append(k)

need_sampled_losses_23 = [[] for _ in range(len(need_sampled_cluster_23))]

for i in range(len(need_sampled_cluster_23)):
    for j in range(len(labels_23)):
        if labels_23[j] == need_sampled_cluster_23[i]:
            need_sampled_losses_23[i].append(label_23_loss[j][0])

for j in range(len(labels_23)):
    if labels_23[j] not in need_sampled_cluster_23:
        selected_sampled_losses_23.append(label_23_loss[j][0])

for loss in need_sampled_losses_23:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_23.append(lo)
print(len(label_23_loss))
print(len(set(selected_sampled_losses_23)))

for loss in list(set(selected_sampled_losses_23)):
    for img in label_23_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第24类的采样=================================================
selected_sampled_losses_24 = []
ms_24 = MeanShift()
ms_24.fit(label_24_loss)

labels_24 = ms_24.labels_  # 第0类中的每个样本对应的簇号

n_clusters_24 = len(np.unique(labels_24))  # 第0类样本有几个簇

need_sampled_cluster_24 = []  # 需要下采样的簇号

cluster_to_num_24 = Counter(labels_24)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_24.keys():
    if cluster_to_num_24[k] > len(labels_24) // n_clusters_24:
        need_sampled_cluster_24.append(k)

need_sampled_losses_24 = [[] for _ in range(len(need_sampled_cluster_24))]

for i in range(len(need_sampled_cluster_24)):
    for j in range(len(labels_24)):
        if labels_24[j] == need_sampled_cluster_24[i]:
            need_sampled_losses_24[i].append(label_24_loss[j][0])

for j in range(len(labels_24)):
    if labels_24[j] not in need_sampled_cluster_24:
        selected_sampled_losses_24.append(label_24_loss[j][0])

for loss in need_sampled_losses_24:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_24.append(lo)
print(len(label_24_loss))
print(len(set(selected_sampled_losses_24)))

for loss in list(set(selected_sampled_losses_24)):
    for img in label_24_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第25类的采样=================================================
selected_sampled_losses_25 = []
ms_25 = MeanShift()
ms_25.fit(label_25_loss)

labels_25 = ms_25.labels_  # 第0类中的每个样本对应的簇号

n_clusters_25 = len(np.unique(labels_25))  # 第0类样本有几个簇

need_sampled_cluster_25 = []  # 需要下采样的簇号

cluster_to_num_25 = Counter(labels_25)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_25.keys():
    if cluster_to_num_25[k] > len(labels_25) // n_clusters_25:
        need_sampled_cluster_25.append(k)

need_sampled_losses_25 = [[] for _ in range(len(need_sampled_cluster_25))]

for i in range(len(need_sampled_cluster_25)):
    for j in range(len(labels_25)):
        if labels_25[j] == need_sampled_cluster_25[i]:
            need_sampled_losses_25[i].append(label_25_loss[j][0])

for j in range(len(labels_25)):
    if labels_25[j] not in need_sampled_cluster_25:
        selected_sampled_losses_25.append(label_25_loss[j][0])

for loss in need_sampled_losses_25:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_25.append(lo)
print(len(label_25_loss))
print(len(set(selected_sampled_losses_25)))

for loss in list(set(selected_sampled_losses_25)):
    for img in label_25_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第26类的采样=================================================
selected_sampled_losses_26 = []
ms_26 = MeanShift()
ms_26.fit(label_26_loss)

labels_26 = ms_26.labels_  # 第0类中的每个样本对应的簇号

n_clusters_26 = len(np.unique(labels_26))  # 第0类样本有几个簇

need_sampled_cluster_26 = []  # 需要下采样的簇号

cluster_to_num_26 = Counter(labels_26)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_26.keys():
    if cluster_to_num_26[k] > len(labels_26) // n_clusters_26:
        need_sampled_cluster_26.append(k)

need_sampled_losses_26 = [[] for _ in range(len(need_sampled_cluster_26))]

for i in range(len(need_sampled_cluster_26)):
    for j in range(len(labels_26)):
        if labels_26[j] == need_sampled_cluster_26[i]:
            need_sampled_losses_26[i].append(label_26_loss[j][0])

for j in range(len(labels_26)):
    if labels_26[j] not in need_sampled_cluster_26:
        selected_sampled_losses_26.append(label_26_loss[j][0])

for loss in need_sampled_losses_26:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_26.append(lo)
print(len(label_26_loss))
print(len(set(selected_sampled_losses_26)))

for loss in list(set(selected_sampled_losses_26)):
    for img in label_26_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第27类的采样=================================================
selected_sampled_losses_27 = []
ms_27 = MeanShift()
ms_27.fit(label_27_loss)

labels_27 = ms_27.labels_  # 第0类中的每个样本对应的簇号

n_clusters_27 = len(np.unique(labels_27))  # 第0类样本有几个簇

need_sampled_cluster_27 = []  # 需要下采样的簇号

cluster_to_num_27 = Counter(labels_27)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_27.keys():
    if cluster_to_num_27[k] > len(labels_27) // n_clusters_27:
        need_sampled_cluster_27.append(k)

need_sampled_losses_27 = [[] for _ in range(len(need_sampled_cluster_27))]

for i in range(len(need_sampled_cluster_27)):
    for j in range(len(labels_27)):
        if labels_27[j] == need_sampled_cluster_27[i]:
            need_sampled_losses_27[i].append(label_27_loss[j][0])

for j in range(len(labels_27)):
    if labels_27[j] not in need_sampled_cluster_27:
        selected_sampled_losses_27.append(label_27_loss[j][0])

for loss in need_sampled_losses_27:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_27.append(lo)
print(len(label_27_loss))
print(len(set(selected_sampled_losses_27)))

for loss in list(set(selected_sampled_losses_27)):
    for img in label_27_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第28类的采样=================================================
selected_sampled_losses_28 = []
ms_28 = MeanShift()
ms_28.fit(label_28_loss)

labels_28 = ms_28.labels_  # 第0类中的每个样本对应的簇号

n_clusters_28 = len(np.unique(labels_28))  # 第0类样本有几个簇

need_sampled_cluster_28 = []  # 需要下采样的簇号

cluster_to_num_28 = Counter(labels_28)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_28.keys():
    if cluster_to_num_28[k] > len(labels_28) // n_clusters_28:
        need_sampled_cluster_28.append(k)

need_sampled_losses_28 = [[] for _ in range(len(need_sampled_cluster_28))]

for i in range(len(need_sampled_cluster_28)):
    for j in range(len(labels_28)):
        if labels_28[j] == need_sampled_cluster_28[i]:
            need_sampled_losses_28[i].append(label_28_loss[j][0])

for j in range(len(labels_28)):
    if labels_28[j] not in need_sampled_cluster_28:
        selected_sampled_losses_28.append(label_28_loss[j][0])

for loss in need_sampled_losses_28:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_28.append(lo)
print(len(label_28_loss))
print(len(set(selected_sampled_losses_28)))

for loss in list(set(selected_sampled_losses_28)):
    for img in label_28_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第29类的采样=================================================
selected_sampled_losses_29 = []
ms_29 = MeanShift()
ms_29.fit(label_29_loss)

labels_29 = ms_29.labels_  # 第0类中的每个样本对应的簇号

n_clusters_29 = len(np.unique(labels_29))  # 第0类样本有几个簇

need_sampled_cluster_29 = []  # 需要下采样的簇号

cluster_to_num_29 = Counter(labels_29)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_29.keys():
    if cluster_to_num_29[k] > len(labels_29) // n_clusters_29:
        need_sampled_cluster_29.append(k)

need_sampled_losses_29 = [[] for _ in range(len(need_sampled_cluster_29))]

for i in range(len(need_sampled_cluster_29)):
    for j in range(len(labels_29)):
        if labels_29[j] == need_sampled_cluster_29[i]:
            need_sampled_losses_29[i].append(label_29_loss[j][0])

for j in range(len(labels_29)):
    if labels_29[j] not in need_sampled_cluster_29:
        selected_sampled_losses_29.append(label_29_loss[j][0])

for loss in need_sampled_losses_29:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_29.append(lo)
print(len(label_29_loss))
print(len(set(selected_sampled_losses_29)))

for loss in list(set(selected_sampled_losses_29)):
    for img in label_29_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第30类的采样=================================================
selected_sampled_losses_30 = []
ms_30 = MeanShift()
ms_30.fit(label_30_loss)

labels_30 = ms_30.labels_  # 第0类中的每个样本对应的簇号

n_clusters_30 = len(np.unique(labels_30))  # 第0类样本有几个簇

need_sampled_cluster_30 = []  # 需要下采样的簇号

cluster_to_num_30 = Counter(labels_30)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_30.keys():
    if cluster_to_num_30[k] > len(labels_30) // n_clusters_30:
        need_sampled_cluster_30.append(k)

need_sampled_losses_30 = [[] for _ in range(len(need_sampled_cluster_30))]

for i in range(len(need_sampled_cluster_30)):
    for j in range(len(labels_30)):
        if labels_30[j] == need_sampled_cluster_30[i]:
            need_sampled_losses_30[i].append(label_30_loss[j][0])

for j in range(len(labels_30)):
    if labels_30[j] not in need_sampled_cluster_30:
        selected_sampled_losses_30.append(label_30_loss[j][0])

for loss in need_sampled_losses_30:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_30.append(lo)
print(len(label_30_loss))
print(len(set(selected_sampled_losses_30)))

for loss in list(set(selected_sampled_losses_30)):
    for img in label_30_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第31类的采样=================================================
selected_sampled_losses_31 = []

ms_31 = MeanShift()
ms_31.fit(label_31_loss)

labels_31 = ms_31.labels_  # 第0类中的每个样本对应的簇号

n_clusters_31 = len(np.unique(labels_31))  # 第0类样本有几个簇

need_sampled_cluster_31 = []  # 需要下采样的簇号

cluster_to_num_31 = Counter(labels_31)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_31.keys():
    if cluster_to_num_31[k] > len(labels_31) // n_clusters_31:
        need_sampled_cluster_31.append(k)

need_sampled_losses_31 = [[] for _ in range(len(need_sampled_cluster_31))]

for i in range(len(need_sampled_cluster_31)):
    for j in range(len(labels_31)):
        if labels_31[j] == need_sampled_cluster_31[i]:
            need_sampled_losses_31[i].append(label_31_loss[j][0])

for j in range(len(labels_31)):
    if labels_31[j] not in need_sampled_cluster_31:
        selected_sampled_losses_31.append(label_31_loss[j][0])

for loss in need_sampled_losses_31:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_31.append(lo)
print(len(label_31_loss))
print(len(set(selected_sampled_losses_31)))

for loss in list(set(selected_sampled_losses_31)):
    for img in label_31_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第32类的采样=================================================
selected_sampled_losses_32 = []
ms_32 = MeanShift()
ms_32.fit(label_32_loss)

labels_32 = ms_32.labels_  # 第0类中的每个样本对应的簇号

n_clusters_32 = len(np.unique(labels_32))  # 第0类样本有几个簇

need_sampled_cluster_32 = []  # 需要下采样的簇号

cluster_to_num_32 = Counter(labels_32)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_32.keys():
    if cluster_to_num_32[k] > len(labels_32) // n_clusters_32:
        need_sampled_cluster_32.append(k)

need_sampled_losses_32 = [[] for _ in range(len(need_sampled_cluster_32))]

for i in range(len(need_sampled_cluster_32)):
    for j in range(len(labels_32)):
        if labels_32[j] == need_sampled_cluster_32[i]:
            need_sampled_losses_32[i].append(label_32_loss[j][0])

for j in range(len(labels_32)):
    if labels_32[j] not in need_sampled_cluster_32:
        selected_sampled_losses_32.append(label_32_loss[j][0])

for loss in need_sampled_losses_32:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_32.append(lo)
print(len(label_32_loss))
print(len(set(selected_sampled_losses_32)))

for loss in list(set(selected_sampled_losses_32)):
    for img in label_32_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第33类的采样=================================================
selected_sampled_losses_33 = []
ms_33 = MeanShift()
ms_33.fit(label_33_loss)

labels_33 = ms_33.labels_  # 第0类中的每个样本对应的簇号

n_clusters_33 = len(np.unique(labels_33))  # 第0类样本有几个簇

need_sampled_cluster_33 = []  # 需要下采样的簇号

cluster_to_num_33 = Counter(labels_33)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_33.keys():
    if cluster_to_num_33[k] > len(labels_33) // n_clusters_33:
        need_sampled_cluster_33.append(k)

need_sampled_losses_33 = [[] for _ in range(len(need_sampled_cluster_33))]

for i in range(len(need_sampled_cluster_33)):
    for j in range(len(labels_33)):
        if labels_33[j] == need_sampled_cluster_33[i]:
            need_sampled_losses_33[i].append(label_33_loss[j][0])

for j in range(len(labels_33)):
    if labels_33[j] not in need_sampled_cluster_33:
        selected_sampled_losses_33.append(label_33_loss[j][0])

for loss in need_sampled_losses_33:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_33.append(lo)
print(len(label_33_loss))
print(len(set(selected_sampled_losses_33)))

for loss in list(set(selected_sampled_losses_33)):
    for img in label_33_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第34类的采样=================================================
selected_sampled_losses_34 = []
ms_34 = MeanShift()
ms_34.fit(label_34_loss)

labels_34 = ms_34.labels_  # 第0类中的每个样本对应的簇号

n_clusters_34 = len(np.unique(labels_34))  # 第0类样本有几个簇

need_sampled_cluster_34 = []  # 需要下采样的簇号

cluster_to_num_34 = Counter(labels_34)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_34.keys():
    if cluster_to_num_34[k] > len(labels_34) // n_clusters_34:
        need_sampled_cluster_34.append(k)

need_sampled_losses_34 = [[] for _ in range(len(need_sampled_cluster_34))]

for i in range(len(need_sampled_cluster_34)):
    for j in range(len(labels_34)):
        if labels_34[j] == need_sampled_cluster_34[i]:
            need_sampled_losses_34[i].append(label_34_loss[j][0])

for j in range(len(labels_34)):
    if labels_34[j] not in need_sampled_cluster_34:
        selected_sampled_losses_34.append(label_34_loss[j][0])

for loss in need_sampled_losses_34:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_34.append(lo)
print(len(label_34_loss))
print(len(set(selected_sampled_losses_34)))

for loss in list(set(selected_sampled_losses_34)):
    for img in label_34_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第35类的采样=================================================
selected_sampled_losses_35 = []
ms_35 = MeanShift()
ms_35.fit(label_35_loss)

labels_35 = ms_35.labels_  # 第0类中的每个样本对应的簇号

n_clusters_35 = len(np.unique(labels_35))  # 第0类样本有几个簇

need_sampled_cluster_35 = []  # 需要下采样的簇号

cluster_to_num_35 = Counter(labels_35)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_35.keys():
    if cluster_to_num_35[k] > len(labels_35) // n_clusters_35:
        need_sampled_cluster_35.append(k)

need_sampled_losses_35 = [[] for _ in range(len(need_sampled_cluster_35))]

for i in range(len(need_sampled_cluster_35)):
    for j in range(len(labels_35)):
        if labels_35[j] == need_sampled_cluster_35[i]:
            need_sampled_losses_35[i].append(label_35_loss[j][0])

for j in range(len(labels_35)):
    if labels_35[j] not in need_sampled_cluster_35:
        selected_sampled_losses_35.append(label_35_loss[j][0])

for loss in need_sampled_losses_35:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_35.append(lo)
print(len(label_35_loss))
print(len(set(selected_sampled_losses_35)))

for loss in list(set(selected_sampled_losses_35)):
    for img in label_35_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第36类的采样=================================================
selected_sampled_losses_36 = []
ms_36 = MeanShift()
ms_36.fit(label_36_loss)

labels_36 = ms_36.labels_  # 第0类中的每个样本对应的簇号

n_clusters_36 = len(np.unique(labels_36))  # 第0类样本有几个簇

need_sampled_cluster_36 = []  # 需要下采样的簇号

cluster_to_num_36 = Counter(labels_36)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_36.keys():
    if cluster_to_num_36[k] > len(labels_36) // n_clusters_36:
        need_sampled_cluster_36.append(k)

need_sampled_losses_36 = [[] for _ in range(len(need_sampled_cluster_36))]

for i in range(len(need_sampled_cluster_36)):
    for j in range(len(labels_36)):
        if labels_36[j] == need_sampled_cluster_36[i]:
            need_sampled_losses_36[i].append(label_36_loss[j][0])

for j in range(len(labels_36)):
    if labels_36[j] not in need_sampled_cluster_36:
        selected_sampled_losses_36.append(label_36_loss[j][0])

for loss in need_sampled_losses_36:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_36.append(lo)
print(len(label_36_loss))
print(len(set(selected_sampled_losses_36)))

for loss in list(set(selected_sampled_losses_36)):
    for img in label_36_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第37类的采样=================================================
selected_sampled_losses_37 = []
ms_37 = MeanShift()
ms_37.fit(label_37_loss)

labels_37 = ms_37.labels_  # 第0类中的每个样本对应的簇号

n_clusters_37 = len(np.unique(labels_37))  # 第0类样本有几个簇

need_sampled_cluster_37 = []  # 需要下采样的簇号

cluster_to_num_37 = Counter(labels_37)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_37.keys():
    if cluster_to_num_37[k] > len(labels_37) // n_clusters_37:
        need_sampled_cluster_37.append(k)

need_sampled_losses_37 = [[] for _ in range(len(need_sampled_cluster_37))]

for i in range(len(need_sampled_cluster_37)):
    for j in range(len(labels_37)):
        if labels_37[j] == need_sampled_cluster_37[i]:
            need_sampled_losses_37[i].append(label_37_loss[j][0])

for j in range(len(labels_37)):
    if labels_37[j] not in need_sampled_cluster_37:
        selected_sampled_losses_37.append(label_37_loss[j][0])

for loss in need_sampled_losses_37:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_37.append(lo)
print(len(label_37_loss))
print(len(set(selected_sampled_losses_37)))

for loss in list(set(selected_sampled_losses_37)):
    for img in label_37_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第38类的采样=================================================
selected_sampled_losses_38 = []
ms_38 = MeanShift()
ms_38.fit(label_38_loss)

labels_38 = ms_38.labels_  # 第0类中的每个样本对应的簇号

n_clusters_38 = len(np.unique(labels_38))  # 第0类样本有几个簇

need_sampled_cluster_38 = []  # 需要下采样的簇号

cluster_to_num_38 = Counter(labels_38)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_38.keys():
    if cluster_to_num_38[k] > len(labels_38) // n_clusters_38:
        need_sampled_cluster_38.append(k)

need_sampled_losses_38 = [[] for _ in range(len(need_sampled_cluster_38))]

for i in range(len(need_sampled_cluster_38)):
    for j in range(len(labels_38)):
        if labels_38[j] == need_sampled_cluster_38[i]:
            need_sampled_losses_38[i].append(label_38_loss[j][0])

for j in range(len(labels_38)):
    if labels_38[j] not in need_sampled_cluster_38:
        selected_sampled_losses_38.append(label_38_loss[j][0])

for loss in need_sampled_losses_38:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_38.append(lo)
print(len(label_38_loss))
print(len(set(selected_sampled_losses_38)))

for loss in list(set(selected_sampled_losses_38)):
    for img in label_38_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第39类的采样=================================================
selected_sampled_losses_39 = []
ms_39 = MeanShift()
ms_39.fit(label_39_loss)

labels_39 = ms_39.labels_  # 第0类中的每个样本对应的簇号

n_clusters_39 = len(np.unique(labels_39))  # 第0类样本有几个簇

need_sampled_cluster_39 = []  # 需要下采样的簇号

cluster_to_num_39 = Counter(labels_39)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_39.keys():
    if cluster_to_num_39[k] > len(labels_39) // n_clusters_39:
        need_sampled_cluster_39.append(k)

need_sampled_losses_39 = [[] for _ in range(len(need_sampled_cluster_39))]

for i in range(len(need_sampled_cluster_39)):
    for j in range(len(labels_39)):
        if labels_39[j] == need_sampled_cluster_39[i]:
            need_sampled_losses_39[i].append(label_39_loss[j][0])

for j in range(len(labels_39)):
    if labels_39[j] not in need_sampled_cluster_39:
        selected_sampled_losses_39.append(label_39_loss[j][0])

for loss in need_sampled_losses_39:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_39.append(lo)
print(len(label_39_loss))
print(len(set(selected_sampled_losses_39)))
for loss in list(set(selected_sampled_losses_39)):
    for img in label_39_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))

# ==================================================针对第40类的采样=================================================

selected_sampled_losses_40 = []

ms_40 = MeanShift()
ms_40.fit(label_40_loss)

labels_40 = ms_40.labels_  # 第0类中的每个样本对应的簇号

n_clusters_40 = len(np.unique(labels_40))  # 第0类样本有几个簇

need_sampled_cluster_40 = []  # 需要下采样的簇号

cluster_to_num_40 = Counter(labels_40)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_40.keys():
    if cluster_to_num_40[k] > len(labels_40) // n_clusters_40:
        need_sampled_cluster_40.append(k)

need_sampled_losses_40 = [[] for _ in range(len(need_sampled_cluster_40))]

for i in range(len(need_sampled_cluster_40)):
    for j in range(len(labels_40)):
        if labels_40[j] == need_sampled_cluster_40[i]:
            need_sampled_losses_40[i].append(label_40_loss[j][0])

for j in range(len(labels_40)):
    if labels_40[j] not in need_sampled_cluster_40:
        selected_sampled_losses_40.append(label_40_loss[j][0])

for loss in need_sampled_losses_40:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_40.append(lo)
print(len(label_40_loss))
print(len(set(selected_sampled_losses_40)))

for loss in list(set(selected_sampled_losses_40)):
    for img in label_40_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第41类的采样=================================================
selected_sampled_losses_41 = []

ms_41 = MeanShift()
ms_41.fit(label_41_loss)

labels_41 = ms_41.labels_  # 第0类中的每个样本对应的簇号

n_clusters_41 = len(np.unique(labels_41))  # 第0类样本有几个簇

need_sampled_cluster_41 = []  # 需要下采样的簇号

cluster_to_num_41 = Counter(labels_41)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_41.keys():
    if cluster_to_num_41[k] > len(labels_41) // n_clusters_41:
        need_sampled_cluster_41.append(k)

need_sampled_losses_41 = [[] for _ in range(len(need_sampled_cluster_41))]

for i in range(len(need_sampled_cluster_41)):
    for j in range(len(labels_41)):
        if labels_41[j] == need_sampled_cluster_41[i]:
            need_sampled_losses_41[i].append(label_41_loss[j][0])

for j in range(len(labels_41)):
    if labels_41[j] not in need_sampled_cluster_41:
        selected_sampled_losses_41.append(label_41_loss[j][0])

for loss in need_sampled_losses_41:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_41.append(lo)
print(len(label_41_loss))
print(len(set(selected_sampled_losses_41)))

for loss in list(set(selected_sampled_losses_41)):
    for img in label_41_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第42类的采样=================================================
selected_sampled_losses_42 = []

ms_42 = MeanShift()
ms_42.fit(label_42_loss)

labels_42 = ms_42.labels_  # 第0类中的每个样本对应的簇号

n_clusters_42 = len(np.unique(labels_42))  # 第0类样本有几个簇

need_sampled_cluster_42 = []  # 需要下采样的簇号

cluster_to_num_42 = Counter(labels_42)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_42.keys():
    if cluster_to_num_42[k] > len(labels_42) // n_clusters_42:
        need_sampled_cluster_42.append(k)

need_sampled_losses_42 = [[] for _ in range(len(need_sampled_cluster_42))]

for i in range(len(need_sampled_cluster_42)):
    for j in range(len(labels_42)):
        if labels_42[j] == need_sampled_cluster_42[i]:
            need_sampled_losses_42[i].append(label_42_loss[j][0])

for j in range(len(labels_42)):
    if labels_42[j] not in need_sampled_cluster_42:
        selected_sampled_losses_42.append(label_42_loss[j][0])

for loss in need_sampled_losses_42:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_42.append(lo)
print(len(label_42_loss))
print(len(set(selected_sampled_losses_42)))

for loss in list(set(selected_sampled_losses_42)):
    for img in label_42_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第43类的采样=================================================
selected_sampled_losses_43 = []

ms_43 = MeanShift()
ms_43.fit(label_43_loss)

labels_43 = ms_43.labels_  # 第0类中的每个样本对应的簇号

n_clusters_43 = len(np.unique(labels_43))  # 第0类样本有几个簇

need_sampled_cluster_43 = []  # 需要下采样的簇号

cluster_to_num_43 = Counter(labels_43)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_43.keys():
    if cluster_to_num_43[k] > len(labels_43) // n_clusters_43:
        need_sampled_cluster_43.append(k)

need_sampled_losses_43 = [[] for _ in range(len(need_sampled_cluster_43))]

for i in range(len(need_sampled_cluster_43)):
    for j in range(len(labels_43)):
        if labels_43[j] == need_sampled_cluster_43[i]:
            need_sampled_losses_43[i].append(label_43_loss[j][0])

for j in range(len(labels_43)):
    if labels_43[j] not in need_sampled_cluster_43:
        selected_sampled_losses_43.append(label_43_loss[j][0])

for loss in need_sampled_losses_43:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_43.append(lo)
print(len(label_43_loss))
print(len(set(selected_sampled_losses_43)))

for loss in list(set(selected_sampled_losses_43)):
    for img in label_43_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第44类的采样=================================================
selected_sampled_losses_44 = []
ms_44 = MeanShift()
ms_44.fit(label_44_loss)

labels_44 = ms_44.labels_  # 第0类中的每个样本对应的簇号

n_clusters_44 = len(np.unique(labels_44))  # 第0类样本有几个簇

need_sampled_cluster_44 = []  # 需要下采样的簇号

cluster_to_num_44 = Counter(labels_44)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_44.keys():
    if cluster_to_num_44[k] > len(labels_44) // n_clusters_44:
        need_sampled_cluster_44.append(k)

need_sampled_losses_44 = [[] for _ in range(len(need_sampled_cluster_44))]

for i in range(len(need_sampled_cluster_44)):
    for j in range(len(labels_44)):
        if labels_44[j] == need_sampled_cluster_44[i]:
            need_sampled_losses_44[i].append(label_44_loss[j][0])

for j in range(len(labels_44)):
    if labels_44[j] not in need_sampled_cluster_44:
        selected_sampled_losses_44.append(label_44_loss[j][0])

for loss in need_sampled_losses_44:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_44.append(lo)
print(len(label_44_loss))
print(len(set(selected_sampled_losses_44)))

for loss in list(set(selected_sampled_losses_44)):
    for img in label_44_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第45类的采样=================================================
selected_sampled_losses_45 = []
ms_45 = MeanShift()
ms_45.fit(label_45_loss)

labels_45 = ms_45.labels_  # 第0类中的每个样本对应的簇号

n_clusters_45 = len(np.unique(labels_45))  # 第0类样本有几个簇

need_sampled_cluster_45 = []  # 需要下采样的簇号

cluster_to_num_45 = Counter(labels_45)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_45.keys():
    if cluster_to_num_45[k] > len(labels_45) // n_clusters_45:
        need_sampled_cluster_45.append(k)

need_sampled_losses_45 = [[] for _ in range(len(need_sampled_cluster_45))]

for i in range(len(need_sampled_cluster_45)):
    for j in range(len(labels_45)):
        if labels_45[j] == need_sampled_cluster_45[i]:
            need_sampled_losses_45[i].append(label_45_loss[j][0])

for j in range(len(labels_45)):
    if labels_45[j] not in need_sampled_cluster_45:
        selected_sampled_losses_45.append(label_45_loss[j][0])

for loss in need_sampled_losses_45:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_45.append(lo)
print(len(label_45_loss))
print(len(set(selected_sampled_losses_45)))

for loss in list(set(selected_sampled_losses_45)):
    for img in label_45_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第46类的采样=================================================
selected_sampled_losses_46 = []
ms_46 = MeanShift()
ms_46.fit(label_46_loss)

labels_46 = ms_46.labels_  # 第0类中的每个样本对应的簇号

n_clusters_46 = len(np.unique(labels_46))  # 第0类样本有几个簇

need_sampled_cluster_46 = []  # 需要下采样的簇号

cluster_to_num_46 = Counter(labels_46)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_46.keys():
    if cluster_to_num_46[k] > len(labels_46) // n_clusters_46:
        need_sampled_cluster_46.append(k)

need_sampled_losses_46 = [[] for _ in range(len(need_sampled_cluster_46))]

for i in range(len(need_sampled_cluster_46)):
    for j in range(len(labels_46)):
        if labels_46[j] == need_sampled_cluster_46[i]:
            need_sampled_losses_46[i].append(label_46_loss[j][0])

for j in range(len(labels_46)):
    if labels_46[j] not in need_sampled_cluster_46:
        selected_sampled_losses_46.append(label_46_loss[j][0])

for loss in need_sampled_losses_46:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_46.append(lo)
print(len(label_46_loss))
print(len(set(selected_sampled_losses_46)))

for loss in list(set(selected_sampled_losses_46)):
    for img in label_46_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第47类的采样=================================================
selected_sampled_losses_47 = []
ms_47 = MeanShift()
ms_47.fit(label_47_loss)

labels_47 = ms_47.labels_  # 第0类中的每个样本对应的簇号

n_clusters_47 = len(np.unique(labels_47))  # 第0类样本有几个簇

need_sampled_cluster_47 = []  # 需要下采样的簇号

cluster_to_num_47 = Counter(labels_47)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_47.keys():
    if cluster_to_num_47[k] > len(labels_47) // n_clusters_47:
        need_sampled_cluster_47.append(k)

need_sampled_losses_47 = [[] for _ in range(len(need_sampled_cluster_47))]

for i in range(len(need_sampled_cluster_47)):
    for j in range(len(labels_47)):
        if labels_47[j] == need_sampled_cluster_47[i]:
            need_sampled_losses_47[i].append(label_47_loss[j][0])

for j in range(len(labels_47)):
    if labels_47[j] not in need_sampled_cluster_47:
        selected_sampled_losses_47.append(label_47_loss[j][0])

for loss in need_sampled_losses_47:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_47.append(lo)
print(len(label_47_loss))
print(len(set(selected_sampled_losses_47)))

for loss in list(set(selected_sampled_losses_47)):
    for img in label_47_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第48类的采样=================================================
selected_sampled_losses_48 = []
ms_48 = MeanShift()
ms_48.fit(label_48_loss)

labels_48 = ms_48.labels_  # 第0类中的每个样本对应的簇号

n_clusters_48 = len(np.unique(labels_48))  # 第0类样本有几个簇

need_sampled_cluster_48 = []  # 需要下采样的簇号

cluster_to_num_48 = Counter(labels_48)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_48.keys():
    if cluster_to_num_48[k] > len(labels_48) // n_clusters_48:
        need_sampled_cluster_48.append(k)

need_sampled_losses_48 = [[] for _ in range(len(need_sampled_cluster_48))]

for i in range(len(need_sampled_cluster_48)):
    for j in range(len(labels_48)):
        if labels_48[j] == need_sampled_cluster_48[i]:
            need_sampled_losses_48[i].append(label_48_loss[j][0])

for j in range(len(labels_48)):
    if labels_48[j] not in need_sampled_cluster_48:
        selected_sampled_losses_48.append(label_48_loss[j][0])

for loss in need_sampled_losses_48:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_48.append(lo)
print(len(label_48_loss))
print(len(set(selected_sampled_losses_48)))

for loss in list(set(selected_sampled_losses_48)):
    for img in label_48_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第49类的采样=================================================
selected_sampled_losses_49 = []
ms_49 = MeanShift()
ms_49.fit(label_49_loss)

labels_49 = ms_49.labels_  # 第0类中的每个样本对应的簇号

n_clusters_49 = len(np.unique(labels_49))  # 第0类样本有几个簇

need_sampled_cluster_49 = []  # 需要下采样的簇号

cluster_to_num_49 = Counter(labels_49)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_49.keys():
    if cluster_to_num_49[k] > len(labels_49) // n_clusters_49:
        need_sampled_cluster_49.append(k)

need_sampled_losses_49 = [[] for _ in range(len(need_sampled_cluster_49))]

for i in range(len(need_sampled_cluster_49)):
    for j in range(len(labels_49)):
        if labels_49[j] == need_sampled_cluster_49[i]:
            need_sampled_losses_49[i].append(label_49_loss[j][0])

for j in range(len(labels_49)):
    if labels_49[j] not in need_sampled_cluster_49:
        selected_sampled_losses_49.append(label_49_loss[j][0])

for loss in need_sampled_losses_49:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_49.append(lo)
print(len(label_49_loss))
print(len(set(selected_sampled_losses_49)))

for loss in list(set(selected_sampled_losses_49)):
    for img in label_49_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第50类的采样=================================================
selected_sampled_losses_50 = []
ms_50 = MeanShift()
ms_50.fit(label_50_loss)

labels_50 = ms_50.labels_  # 第0类中的每个样本对应的簇号

n_clusters_50 = len(np.unique(labels_50))  # 第0类样本有几个簇

need_sampled_cluster_50 = []  # 需要下采样的簇号

cluster_to_num_50 = Counter(labels_50)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_50.keys():
    if cluster_to_num_50[k] > len(labels_50) // n_clusters_50:
        need_sampled_cluster_50.append(k)

need_sampled_losses_50 = [[] for _ in range(len(need_sampled_cluster_50))]

for i in range(len(need_sampled_cluster_50)):
    for j in range(len(labels_50)):
        if labels_50[j] == need_sampled_cluster_50[i]:
            need_sampled_losses_50[i].append(label_50_loss[j][0])

for j in range(len(labels_50)):
    if labels_50[j] not in need_sampled_cluster_50:
        selected_sampled_losses_50.append(label_50_loss[j][0])

for loss in need_sampled_losses_50:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_50.append(lo)
print(len(label_50_loss))
print(len(set(selected_sampled_losses_50)))

for loss in list(set(selected_sampled_losses_50)):
    for img in label_50_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第51类的采样=================================================
selected_sampled_losses_51 = []

ms_51 = MeanShift()
ms_51.fit(label_51_loss)

labels_51 = ms_51.labels_  # 第0类中的每个样本对应的簇号

n_clusters_51 = len(np.unique(labels_51))  # 第0类样本有几个簇

need_sampled_cluster_51 = []  # 需要下采样的簇号

cluster_to_num_51 = Counter(labels_51)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_51.keys():
    if cluster_to_num_51[k] > len(labels_51) // n_clusters_51:
        need_sampled_cluster_51.append(k)

need_sampled_losses_51 = [[] for _ in range(len(need_sampled_cluster_51))]

for i in range(len(need_sampled_cluster_51)):
    for j in range(len(labels_51)):
        if labels_51[j] == need_sampled_cluster_51[i]:
            need_sampled_losses_51[i].append(label_51_loss[j][0])

for j in range(len(labels_51)):
    if labels_51[j] not in need_sampled_cluster_51:
        selected_sampled_losses_51.append(label_51_loss[j][0])

for loss in need_sampled_losses_51:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_51.append(lo)
print(len(label_51_loss))
print(len(set(selected_sampled_losses_51)))

for loss in list(set(selected_sampled_losses_51)):
    for img in label_51_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第52类的采样=================================================
selected_sampled_losses_52 = []
ms_52 = MeanShift()
ms_52.fit(label_52_loss)

labels_52 = ms_52.labels_  # 第0类中的每个样本对应的簇号

n_clusters_52 = len(np.unique(labels_52))  # 第0类样本有几个簇

need_sampled_cluster_52 = []  # 需要下采样的簇号

cluster_to_num_52 = Counter(labels_52)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_52.keys():
    if cluster_to_num_52[k] > len(labels_52) // n_clusters_52:
        need_sampled_cluster_52.append(k)

need_sampled_losses_52 = [[] for _ in range(len(need_sampled_cluster_52))]

for i in range(len(need_sampled_cluster_52)):
    for j in range(len(labels_52)):
        if labels_52[j] == need_sampled_cluster_52[i]:
            need_sampled_losses_52[i].append(label_52_loss[j][0])

for j in range(len(labels_52)):
    if labels_52[j] not in need_sampled_cluster_52:
        selected_sampled_losses_52.append(label_52_loss[j][0])

for loss in need_sampled_losses_52:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_52.append(lo)
print(len(label_52_loss))
print(len(set(selected_sampled_losses_52)))

for loss in list(set(selected_sampled_losses_52)):
    for img in label_52_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第53类的采样=================================================
selected_sampled_losses_53 = []
ms_53 = MeanShift()
ms_53.fit(label_53_loss)

labels_53 = ms_53.labels_  # 第0类中的每个样本对应的簇号

n_clusters_53 = len(np.unique(labels_53))  # 第0类样本有几个簇

need_sampled_cluster_53 = []  # 需要下采样的簇号

cluster_to_num_53 = Counter(labels_53)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_53.keys():
    if cluster_to_num_53[k] > len(labels_53) // n_clusters_53:
        need_sampled_cluster_53.append(k)

need_sampled_losses_53 = [[] for _ in range(len(need_sampled_cluster_53))]

for i in range(len(need_sampled_cluster_53)):
    for j in range(len(labels_53)):
        if labels_53[j] == need_sampled_cluster_53[i]:
            need_sampled_losses_53[i].append(label_53_loss[j][0])

for j in range(len(labels_53)):
    if labels_53[j] not in need_sampled_cluster_53:
        selected_sampled_losses_53.append(label_53_loss[j][0])

for loss in need_sampled_losses_53:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_53.append(lo)
print(len(label_53_loss))
print(len(set(selected_sampled_losses_53)))

for loss in list(set(selected_sampled_losses_53)):
    for img in label_53_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第54类的采样=================================================
selected_sampled_losses_54 = []
ms_54 = MeanShift()
ms_54.fit(label_54_loss)

labels_54 = ms_54.labels_  # 第0类中的每个样本对应的簇号

n_clusters_54 = len(np.unique(labels_54))  # 第0类样本有几个簇

need_sampled_cluster_54 = []  # 需要下采样的簇号

cluster_to_num_54 = Counter(labels_54)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_54.keys():
    if cluster_to_num_54[k] > len(labels_54) // n_clusters_54:
        need_sampled_cluster_54.append(k)

need_sampled_losses_54 = [[] for _ in range(len(need_sampled_cluster_54))]

for i in range(len(need_sampled_cluster_54)):
    for j in range(len(labels_54)):
        if labels_54[j] == need_sampled_cluster_54[i]:
            need_sampled_losses_54[i].append(label_54_loss[j][0])

for j in range(len(labels_54)):
    if labels_54[j] not in need_sampled_cluster_54:
        selected_sampled_losses_54.append(label_54_loss[j][0])

for loss in need_sampled_losses_54:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_54.append(lo)
print(len(label_54_loss))
print(len(set(selected_sampled_losses_54)))

for loss in list(set(selected_sampled_losses_54)):
    for img in label_54_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第55类的采样=================================================
selected_sampled_losses_55 = []
ms_55 = MeanShift()
ms_55.fit(label_55_loss)

labels_55 = ms_55.labels_  # 第0类中的每个样本对应的簇号

n_clusters_55 = len(np.unique(labels_55))  # 第0类样本有几个簇

need_sampled_cluster_55 = []  # 需要下采样的簇号

cluster_to_num_55 = Counter(labels_55)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_55.keys():
    if cluster_to_num_55[k] > len(labels_55) // n_clusters_55:
        need_sampled_cluster_55.append(k)

need_sampled_losses_55 = [[] for _ in range(len(need_sampled_cluster_55))]

for i in range(len(need_sampled_cluster_55)):
    for j in range(len(labels_55)):
        if labels_55[j] == need_sampled_cluster_55[i]:
            need_sampled_losses_55[i].append(label_55_loss[j][0])

for j in range(len(labels_55)):
    if labels_55[j] not in need_sampled_cluster_55:
        selected_sampled_losses_55.append(label_55_loss[j][0])

for loss in need_sampled_losses_55:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_55.append(lo)
print(len(label_55_loss))
print(len(set(selected_sampled_losses_55)))

for loss in list(set(selected_sampled_losses_55)):
    for img in label_55_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第56类的采样=================================================
selected_sampled_losses_56 = []
ms_56 = MeanShift()
ms_56.fit(label_56_loss)

labels_56 = ms_56.labels_  # 第0类中的每个样本对应的簇号

n_clusters_56 = len(np.unique(labels_56))  # 第0类样本有几个簇

need_sampled_cluster_56 = []  # 需要下采样的簇号

cluster_to_num_56 = Counter(labels_56)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_56.keys():
    if cluster_to_num_56[k] > len(labels_56) // n_clusters_56:
        need_sampled_cluster_56.append(k)

need_sampled_losses_56 = [[] for _ in range(len(need_sampled_cluster_56))]

for i in range(len(need_sampled_cluster_56)):
    for j in range(len(labels_56)):
        if labels_56[j] == need_sampled_cluster_56[i]:
            need_sampled_losses_56[i].append(label_56_loss[j][0])

for j in range(len(labels_56)):
    if labels_56[j] not in need_sampled_cluster_56:
        selected_sampled_losses_56.append(label_56_loss[j][0])

for loss in need_sampled_losses_56:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_56.append(lo)
print(len(label_56_loss))
print(len(set(selected_sampled_losses_56)))

for loss in list(set(selected_sampled_losses_56)):
    for img in label_56_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第57类的采样=================================================
selected_sampled_losses_57 = []
ms_57 = MeanShift()
ms_57.fit(label_57_loss)

labels_57 = ms_57.labels_  # 第0类中的每个样本对应的簇号

n_clusters_57 = len(np.unique(labels_57))  # 第0类样本有几个簇

need_sampled_cluster_57 = []  # 需要下采样的簇号

cluster_to_num_57 = Counter(labels_57)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_57.keys():
    if cluster_to_num_57[k] > len(labels_57) // n_clusters_57:
        need_sampled_cluster_57.append(k)

need_sampled_losses_57 = [[] for _ in range(len(need_sampled_cluster_57))]

for i in range(len(need_sampled_cluster_57)):
    for j in range(len(labels_57)):
        if labels_57[j] == need_sampled_cluster_57[i]:
            need_sampled_losses_57[i].append(label_57_loss[j][0])

for j in range(len(labels_57)):
    if labels_57[j] not in need_sampled_cluster_57:
        selected_sampled_losses_57.append(label_57_loss[j][0])

for loss in need_sampled_losses_57:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_57.append(lo)
print(len(label_57_loss))
print(len(set(selected_sampled_losses_57)))

for loss in list(set(selected_sampled_losses_57)):
    for img in label_57_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第58类的采样=================================================
selected_sampled_losses_58 = []
ms_58 = MeanShift()
ms_58.fit(label_58_loss)

labels_58 = ms_58.labels_  # 第0类中的每个样本对应的簇号

n_clusters_58 = len(np.unique(labels_58))  # 第0类样本有几个簇

need_sampled_cluster_58 = []  # 需要下采样的簇号

cluster_to_num_58 = Counter(labels_58)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_58.keys():
    if cluster_to_num_58[k] > len(labels_58) // n_clusters_58:
        need_sampled_cluster_58.append(k)

need_sampled_losses_58 = [[] for _ in range(len(need_sampled_cluster_58))]

for i in range(len(need_sampled_cluster_58)):
    for j in range(len(labels_58)):
        if labels_58[j] == need_sampled_cluster_58[i]:
            need_sampled_losses_58[i].append(label_58_loss[j][0])

for j in range(len(labels_58)):
    if labels_58[j] not in need_sampled_cluster_58:
        selected_sampled_losses_58.append(label_58_loss[j][0])

for loss in need_sampled_losses_58:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_58.append(lo)
print(len(label_58_loss))
print(len(set(selected_sampled_losses_58)))

for loss in list(set(selected_sampled_losses_58)):
    for img in label_58_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第59类的采样=================================================
selected_sampled_losses_59 = []
ms_59 = MeanShift()
ms_59.fit(label_59_loss)

labels_59 = ms_59.labels_  # 第0类中的每个样本对应的簇号

n_clusters_59 = len(np.unique(labels_59))  # 第0类样本有几个簇

need_sampled_cluster_59 = []  # 需要下采样的簇号

cluster_to_num_59 = Counter(labels_59)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_59.keys():
    if cluster_to_num_59[k] > len(labels_59) // n_clusters_59:
        need_sampled_cluster_59.append(k)

need_sampled_losses_59 = [[] for _ in range(len(need_sampled_cluster_59))]

for i in range(len(need_sampled_cluster_59)):
    for j in range(len(labels_59)):
        if labels_59[j] == need_sampled_cluster_59[i]:
            need_sampled_losses_59[i].append(label_59_loss[j][0])

for j in range(len(labels_59)):
    if labels_59[j] not in need_sampled_cluster_59:
        selected_sampled_losses_59.append(label_59_loss[j][0])

for loss in need_sampled_losses_59:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_59.append(lo)
print(len(label_59_loss))
print(len(set(selected_sampled_losses_59)))
for loss in list(set(selected_sampled_losses_59)):
    for img in label_59_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))

# ==================================================针对第60类的采样=================================================

selected_sampled_losses_60 = []

ms_60 = MeanShift()
ms_60.fit(label_60_loss)

labels_60 = ms_60.labels_  # 第_60类中的每个样本对应的簇号

n_clusters_60 = len(np.unique(labels_60))  # 第_60类样本有几个簇

need_sampled_cluster_60 = []  # 需要下采样的簇号

cluster_to_num_60 = Counter(labels_60)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_60.keys():
    if cluster_to_num_60[k] > len(labels_60) // n_clusters_60:
        need_sampled_cluster_60.append(k)

need_sampled_losses_60 = [[] for _ in range(len(need_sampled_cluster_60))]

for i in range(len(need_sampled_cluster_60)):
    for j in range(len(labels_60)):
        if labels_60[j] == need_sampled_cluster_60[i]:
            need_sampled_losses_60[i].append(label_60_loss[j][0])

for j in range(len(labels_60)):
    if labels_60[j] not in need_sampled_cluster_60:
        selected_sampled_losses_60.append(label_60_loss[j][0])

for loss in need_sampled_losses_60:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_60.append(lo)
print(len(label_60_loss))
print(len(set(selected_sampled_losses_60)))

for loss in list(set(selected_sampled_losses_60)):
    for img in label_60_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第61类的采样=================================================
selected_sampled_losses_61 = []

ms_61 = MeanShift()
ms_61.fit(label_61_loss)

labels_61 = ms_61.labels_  # 第0类中的每个样本对应的簇号

n_clusters_61 = len(np.unique(labels_61))  # 第0类样本有几个簇

need_sampled_cluster_61 = []  # 需要下采样的簇号

cluster_to_num_61 = Counter(labels_61)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_61.keys():
    if cluster_to_num_61[k] > len(labels_61) // n_clusters_61:
        need_sampled_cluster_61.append(k)

need_sampled_losses_61 = [[] for _ in range(len(need_sampled_cluster_61))]

for i in range(len(need_sampled_cluster_61)):
    for j in range(len(labels_61)):
        if labels_61[j] == need_sampled_cluster_61[i]:
            need_sampled_losses_61[i].append(label_61_loss[j][0])

for j in range(len(labels_61)):
    if labels_61[j] not in need_sampled_cluster_61:
        selected_sampled_losses_61.append(label_61_loss[j][0])

for loss in need_sampled_losses_61:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_61.append(lo)
print(len(label_61_loss))
print(len(set(selected_sampled_losses_61)))

for loss in list(set(selected_sampled_losses_61)):
    for img in label_61_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第62类的采样=================================================
selected_sampled_losses_62 = []

ms_62 = MeanShift()
ms_62.fit(label_62_loss)

labels_62 = ms_62.labels_  # 第0类中的每个样本对应的簇号

n_clusters_62 = len(np.unique(labels_62))  # 第0类样本有几个簇

need_sampled_cluster_62 = []  # 需要下采样的簇号

cluster_to_num_62 = Counter(labels_62)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_62.keys():
    if cluster_to_num_62[k] > len(labels_62) // n_clusters_62:
        need_sampled_cluster_62.append(k)

need_sampled_losses_62 = [[] for _ in range(len(need_sampled_cluster_62))]

for i in range(len(need_sampled_cluster_62)):
    for j in range(len(labels_62)):
        if labels_62[j] == need_sampled_cluster_62[i]:
            need_sampled_losses_62[i].append(label_62_loss[j][0])

for j in range(len(labels_62)):
    if labels_62[j] not in need_sampled_cluster_62:
        selected_sampled_losses_62.append(label_62_loss[j][0])

for loss in need_sampled_losses_62:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_62.append(lo)
print(len(label_62_loss))
print(len(set(selected_sampled_losses_62)))

for loss in list(set(selected_sampled_losses_62)):
    for img in label_62_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第63类的采样=================================================
selected_sampled_losses_63 = []

ms_63 = MeanShift()
ms_63.fit(label_63_loss)

labels_63 = ms_63.labels_  # 第0类中的每个样本对应的簇号

n_clusters_63 = len(np.unique(labels_63))  # 第0类样本有几个簇

need_sampled_cluster_63 = []  # 需要下采样的簇号

cluster_to_num_63 = Counter(labels_63)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_63.keys():
    if cluster_to_num_63[k] > len(labels_63) // n_clusters_63:
        need_sampled_cluster_63.append(k)

need_sampled_losses_63 = [[] for _ in range(len(need_sampled_cluster_63))]

for i in range(len(need_sampled_cluster_63)):
    for j in range(len(labels_63)):
        if labels_63[j] == need_sampled_cluster_63[i]:
            need_sampled_losses_63[i].append(label_63_loss[j][0])

for j in range(len(labels_63)):
    if labels_63[j] not in need_sampled_cluster_63:
        selected_sampled_losses_63.append(label_63_loss[j][0])

for loss in need_sampled_losses_63:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_63.append(lo)
print(len(label_63_loss))
print(len(set(selected_sampled_losses_63)))

for loss in list(set(selected_sampled_losses_63)):
    for img in label_63_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第64类的采样=================================================
selected_sampled_losses_64 = []
ms_64 = MeanShift()
ms_64.fit(label_64_loss)

labels_64 = ms_64.labels_  # 第0类中的每个样本对应的簇号

n_clusters_64 = len(np.unique(labels_64))  # 第0类样本有几个簇

need_sampled_cluster_64 = []  # 需要下采样的簇号

cluster_to_num_64 = Counter(labels_64)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_64.keys():
    if cluster_to_num_64[k] > len(labels_64) // n_clusters_64:
        need_sampled_cluster_64.append(k)

need_sampled_losses_64 = [[] for _ in range(len(need_sampled_cluster_64))]

for i in range(len(need_sampled_cluster_64)):
    for j in range(len(labels_64)):
        if labels_64[j] == need_sampled_cluster_64[i]:
            need_sampled_losses_64[i].append(label_64_loss[j][0])

for j in range(len(labels_64)):
    if labels_64[j] not in need_sampled_cluster_64:
        selected_sampled_losses_64.append(label_64_loss[j][0])

for loss in need_sampled_losses_64:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_64.append(lo)
print(len(label_64_loss))
print(len(set(selected_sampled_losses_64)))

for loss in list(set(selected_sampled_losses_64)):
    for img in label_64_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第65类的采样=================================================
selected_sampled_losses_65 = []
ms_65 = MeanShift()
ms_65.fit(label_65_loss)

labels_65 = ms_65.labels_  # 第0类中的每个样本对应的簇号

n_clusters_65 = len(np.unique(labels_65))  # 第0类样本有几个簇

need_sampled_cluster_65 = []  # 需要下采样的簇号

cluster_to_num_65 = Counter(labels_65)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_65.keys():
    if cluster_to_num_65[k] > len(labels_65) // n_clusters_65:
        need_sampled_cluster_65.append(k)

need_sampled_losses_65 = [[] for _ in range(len(need_sampled_cluster_65))]

for i in range(len(need_sampled_cluster_65)):
    for j in range(len(labels_65)):
        if labels_65[j] == need_sampled_cluster_65[i]:
            need_sampled_losses_65[i].append(label_65_loss[j][0])

for j in range(len(labels_65)):
    if labels_65[j] not in need_sampled_cluster_65:
        selected_sampled_losses_65.append(label_65_loss[j][0])

for loss in need_sampled_losses_65:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_65.append(lo)
print(len(label_65_loss))
print(len(set(selected_sampled_losses_65)))

for loss in list(set(selected_sampled_losses_65)):
    for img in label_65_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第66类的采样=================================================
selected_sampled_losses_66 = []
ms_66 = MeanShift()
ms_66.fit(label_66_loss)

labels_66 = ms_66.labels_  # 第0类中的每个样本对应的簇号

n_clusters_66 = len(np.unique(labels_66))  # 第0类样本有几个簇

need_sampled_cluster_66 = []  # 需要下采样的簇号

cluster_to_num_66 = Counter(labels_66)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_66.keys():
    if cluster_to_num_66[k] > len(labels_66) // n_clusters_66:
        need_sampled_cluster_66.append(k)

need_sampled_losses_66 = [[] for _ in range(len(need_sampled_cluster_66))]

for i in range(len(need_sampled_cluster_66)):
    for j in range(len(labels_66)):
        if labels_66[j] == need_sampled_cluster_66[i]:
            need_sampled_losses_66[i].append(label_66_loss[j][0])

for j in range(len(labels_66)):
    if labels_66[j] not in need_sampled_cluster_66:
        selected_sampled_losses_66.append(label_66_loss[j][0])

for loss in need_sampled_losses_66:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_66.append(lo)
print(len(label_66_loss))
print(len(set(selected_sampled_losses_66)))

for loss in list(set(selected_sampled_losses_66)):
    for img in label_66_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第67类的采样=================================================
selected_sampled_losses_67 = []
ms_67 = MeanShift()
ms_67.fit(label_67_loss)

labels_67 = ms_67.labels_  # 第0类中的每个样本对应的簇号

n_clusters_67 = len(np.unique(labels_67))  # 第0类样本有几个簇

need_sampled_cluster_67 = []  # 需要下采样的簇号

cluster_to_num_67 = Counter(labels_67)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_67.keys():
    if cluster_to_num_67[k] > len(labels_67) // n_clusters_67:
        need_sampled_cluster_67.append(k)

need_sampled_losses_67 = [[] for _ in range(len(need_sampled_cluster_67))]

for i in range(len(need_sampled_cluster_67)):
    for j in range(len(labels_67)):
        if labels_67[j] == need_sampled_cluster_67[i]:
            need_sampled_losses_67[i].append(label_67_loss[j][0])

for j in range(len(labels_67)):
    if labels_67[j] not in need_sampled_cluster_67:
        selected_sampled_losses_67.append(label_67_loss[j][0])

for loss in need_sampled_losses_67:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_67.append(lo)
print(len(label_67_loss))
print(len(set(selected_sampled_losses_67)))

for loss in list(set(selected_sampled_losses_67)):
    for img in label_67_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第68类的采样=================================================
selected_sampled_losses_68 = []
ms_68 = MeanShift()
ms_68.fit(label_68_loss)

labels_68 = ms_68.labels_  # 第0类中的每个样本对应的簇号

n_clusters_68 = len(np.unique(labels_68))  # 第0类样本有几个簇

need_sampled_cluster_68 = []  # 需要下采样的簇号

cluster_to_num_68 = Counter(labels_68)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_68.keys():
    if cluster_to_num_68[k] > len(labels_68) // n_clusters_68:
        need_sampled_cluster_68.append(k)

need_sampled_losses_68 = [[] for _ in range(len(need_sampled_cluster_68))]

for i in range(len(need_sampled_cluster_68)):
    for j in range(len(labels_68)):
        if labels_68[j] == need_sampled_cluster_68[i]:
            need_sampled_losses_68[i].append(label_68_loss[j][0])

for j in range(len(labels_68)):
    if labels_68[j] not in need_sampled_cluster_68:
        selected_sampled_losses_68.append(label_68_loss[j][0])

for loss in need_sampled_losses_68:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_68.append(lo)
print(len(label_68_loss))
print(len(set(selected_sampled_losses_68)))

for loss in list(set(selected_sampled_losses_68)):
    for img in label_68_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第69类的采样=================================================
selected_sampled_losses_69 = []
ms_69 = MeanShift()
ms_69.fit(label_69_loss)

labels_69 = ms_69.labels_  # 第0类中的每个样本对应的簇号

n_clusters_69 = len(np.unique(labels_69))  # 第0类样本有几个簇

need_sampled_cluster_69 = []  # 需要下采样的簇号

cluster_to_num_69 = Counter(labels_69)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_69.keys():
    if cluster_to_num_69[k] > len(labels_69) // n_clusters_69:
        need_sampled_cluster_69.append(k)

need_sampled_losses_69 = [[] for _ in range(len(need_sampled_cluster_69))]

for i in range(len(need_sampled_cluster_69)):
    for j in range(len(labels_69)):
        if labels_69[j] == need_sampled_cluster_69[i]:
            need_sampled_losses_69[i].append(label_69_loss[j][0])

for j in range(len(labels_69)):
    if labels_69[j] not in need_sampled_cluster_69:
        selected_sampled_losses_69.append(label_69_loss[j][0])

for loss in need_sampled_losses_69:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_69.append(lo)
print(len(label_69_loss))
print(len(set(selected_sampled_losses_69)))

for loss in list(set(selected_sampled_losses_69)):
    for img in label_69_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第70类的采样=================================================
selected_sampled_losses_70 = []
ms_70 = MeanShift()
ms_70.fit(label_70_loss)

labels_70 = ms_70.labels_  # 第0类中的每个样本对应的簇号

n_clusters_70 = len(np.unique(labels_70))  # 第0类样本有几个簇

need_sampled_cluster_70 = []  # 需要下采样的簇号

cluster_to_num_70 = Counter(labels_70)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_70.keys():
    if cluster_to_num_70[k] > len(labels_70) // n_clusters_70:
        need_sampled_cluster_70.append(k)

need_sampled_losses_70 = [[] for _ in range(len(need_sampled_cluster_70))]

for i in range(len(need_sampled_cluster_70)):
    for j in range(len(labels_70)):
        if labels_70[j] == need_sampled_cluster_70[i]:
            need_sampled_losses_70[i].append(label_70_loss[j][0])

for j in range(len(labels_70)):
    if labels_70[j] not in need_sampled_cluster_70:
        selected_sampled_losses_70.append(label_70_loss[j][0])

for loss in need_sampled_losses_70:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_70.append(lo)
print(len(label_70_loss))
print(len(set(selected_sampled_losses_70)))

for loss in list(set(selected_sampled_losses_70)):
    for img in label_70_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第71类的采样=================================================
selected_sampled_losses_71 = []

ms_71 = MeanShift()
ms_71.fit(label_71_loss)

labels_71 = ms_71.labels_  # 第0类中的每个样本对应的簇号

n_clusters_71 = len(np.unique(labels_71))  # 第0类样本有几个簇

need_sampled_cluster_71 = []  # 需要下采样的簇号

cluster_to_num_71 = Counter(labels_71)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_71.keys():
    if cluster_to_num_71[k] > len(labels_71) // n_clusters_71:
        need_sampled_cluster_71.append(k)

need_sampled_losses_71 = [[] for _ in range(len(need_sampled_cluster_71))]

for i in range(len(need_sampled_cluster_71)):
    for j in range(len(labels_71)):
        if labels_71[j] == need_sampled_cluster_71[i]:
            need_sampled_losses_71[i].append(label_71_loss[j][0])

for j in range(len(labels_71)):
    if labels_71[j] not in need_sampled_cluster_71:
        selected_sampled_losses_71.append(label_71_loss[j][0])

for loss in need_sampled_losses_71:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_71.append(lo)
print(len(label_71_loss))
print(len(set(selected_sampled_losses_71)))

for loss in list(set(selected_sampled_losses_71)):
    for img in label_71_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第72类的采样=================================================
selected_sampled_losses_72 = []
ms_72 = MeanShift()
ms_72.fit(label_72_loss)

labels_72 = ms_72.labels_  # 第0类中的每个样本对应的簇号

n_clusters_72 = len(np.unique(labels_72))  # 第0类样本有几个簇

need_sampled_cluster_72 = []  # 需要下采样的簇号

cluster_to_num_72 = Counter(labels_72)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_72.keys():
    if cluster_to_num_72[k] > len(labels_72) // n_clusters_72:
        need_sampled_cluster_72.append(k)

need_sampled_losses_72 = [[] for _ in range(len(need_sampled_cluster_72))]

for i in range(len(need_sampled_cluster_72)):
    for j in range(len(labels_72)):
        if labels_72[j] == need_sampled_cluster_72[i]:
            need_sampled_losses_72[i].append(label_72_loss[j][0])

for j in range(len(labels_72)):
    if labels_72[j] not in need_sampled_cluster_72:
        selected_sampled_losses_72.append(label_72_loss[j][0])

for loss in need_sampled_losses_72:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_72.append(lo)
print(len(label_72_loss))
print(len(set(selected_sampled_losses_72)))

for loss in list(set(selected_sampled_losses_72)):
    for img in label_72_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第73类的采样=================================================
selected_sampled_losses_73 = []
ms_73 = MeanShift()
ms_73.fit(label_73_loss)

labels_73 = ms_73.labels_  # 第0类中的每个样本对应的簇号

n_clusters_73 = len(np.unique(labels_73))  # 第0类样本有几个簇

need_sampled_cluster_73 = []  # 需要下采样的簇号

cluster_to_num_73 = Counter(labels_73)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_73.keys():
    if cluster_to_num_73[k] > len(labels_73) // n_clusters_73:
        need_sampled_cluster_73.append(k)

need_sampled_losses_73 = [[] for _ in range(len(need_sampled_cluster_73))]

for i in range(len(need_sampled_cluster_73)):
    for j in range(len(labels_73)):
        if labels_73[j] == need_sampled_cluster_73[i]:
            need_sampled_losses_73[i].append(label_73_loss[j][0])

for j in range(len(labels_73)):
    if labels_73[j] not in need_sampled_cluster_73:
        selected_sampled_losses_73.append(label_73_loss[j][0])

for loss in need_sampled_losses_73:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_73.append(lo)
print(len(label_73_loss))
print(len(set(selected_sampled_losses_73)))

for loss in list(set(selected_sampled_losses_73)):
    for img in label_73_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第74类的采样=================================================
selected_sampled_losses_74 = []
ms_74 = MeanShift()
ms_74.fit(label_74_loss)

labels_74 = ms_74.labels_  # 第0类中的每个样本对应的簇号

n_clusters_74 = len(np.unique(labels_74))  # 第0类样本有几个簇

need_sampled_cluster_74 = []  # 需要下采样的簇号

cluster_to_num_74 = Counter(labels_74)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_74.keys():
    if cluster_to_num_74[k] > len(labels_74) // n_clusters_74:
        need_sampled_cluster_74.append(k)

need_sampled_losses_74 = [[] for _ in range(len(need_sampled_cluster_74))]

for i in range(len(need_sampled_cluster_74)):
    for j in range(len(labels_74)):
        if labels_74[j] == need_sampled_cluster_74[i]:
            need_sampled_losses_74[i].append(label_74_loss[j][0])

for j in range(len(labels_74)):
    if labels_74[j] not in need_sampled_cluster_74:
        selected_sampled_losses_74.append(label_74_loss[j][0])

for loss in need_sampled_losses_74:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_74.append(lo)
print(len(label_74_loss))
print(len(set(selected_sampled_losses_74)))

for loss in list(set(selected_sampled_losses_74)):
    for img in label_74_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第75类的采样=================================================
selected_sampled_losses_75 = []
ms_75 = MeanShift()
ms_75.fit(label_75_loss)

labels_75 = ms_75.labels_  # 第0类中的每个样本对应的簇号

n_clusters_75 = len(np.unique(labels_75))  # 第0类样本有几个簇

need_sampled_cluster_75 = []  # 需要下采样的簇号

cluster_to_num_75 = Counter(labels_75)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_75.keys():
    if cluster_to_num_75[k] > len(labels_75) // n_clusters_75:
        need_sampled_cluster_75.append(k)

need_sampled_losses_75 = [[] for _ in range(len(need_sampled_cluster_75))]

for i in range(len(need_sampled_cluster_75)):
    for j in range(len(labels_75)):
        if labels_75[j] == need_sampled_cluster_75[i]:
            need_sampled_losses_75[i].append(label_75_loss[j][0])

for j in range(len(labels_75)):
    if labels_75[j] not in need_sampled_cluster_75:
        selected_sampled_losses_75.append(label_75_loss[j][0])

for loss in need_sampled_losses_75:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_75.append(lo)
print(len(label_75_loss))
print(len(set(selected_sampled_losses_75)))

for loss in list(set(selected_sampled_losses_75)):
    for img in label_75_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第76类的采样=================================================
selected_sampled_losses_76 = []
ms_76 = MeanShift()
ms_76.fit(label_76_loss)

labels_76 = ms_76.labels_  # 第0类中的每个样本对应的簇号

n_clusters_76 = len(np.unique(labels_76))  # 第0类样本有几个簇

need_sampled_cluster_76 = []  # 需要下采样的簇号

cluster_to_num_76 = Counter(labels_76)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_76.keys():
    if cluster_to_num_76[k] > len(labels_76) // n_clusters_76:
        need_sampled_cluster_76.append(k)

need_sampled_losses_76 = [[] for _ in range(len(need_sampled_cluster_76))]

for i in range(len(need_sampled_cluster_76)):
    for j in range(len(labels_76)):
        if labels_76[j] == need_sampled_cluster_76[i]:
            need_sampled_losses_76[i].append(label_76_loss[j][0])

for j in range(len(labels_76)):
    if labels_76[j] not in need_sampled_cluster_76:
        selected_sampled_losses_76.append(label_76_loss[j][0])

for loss in need_sampled_losses_76:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_76.append(lo)
print(len(label_76_loss))
print(len(set(selected_sampled_losses_76)))

for loss in list(set(selected_sampled_losses_76)):
    for img in label_76_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第77类的采样=================================================
selected_sampled_losses_77 = []
ms_77 = MeanShift()
ms_77.fit(label_77_loss)

labels_77 = ms_77.labels_  # 第0类中的每个样本对应的簇号

n_clusters_77 = len(np.unique(labels_77))  # 第0类样本有几个簇

need_sampled_cluster_77 = []  # 需要下采样的簇号

cluster_to_num_77 = Counter(labels_77)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_77.keys():
    if cluster_to_num_77[k] > len(labels_77) // n_clusters_77:
        need_sampled_cluster_77.append(k)

need_sampled_losses_77 = [[] for _ in range(len(need_sampled_cluster_77))]

for i in range(len(need_sampled_cluster_77)):
    for j in range(len(labels_77)):
        if labels_77[j] == need_sampled_cluster_77[i]:
            need_sampled_losses_77[i].append(label_77_loss[j][0])

for j in range(len(labels_77)):
    if labels_77[j] not in need_sampled_cluster_77:
        selected_sampled_losses_77.append(label_77_loss[j][0])

for loss in need_sampled_losses_77:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_77.append(lo)
print(len(label_77_loss))
print(len(set(selected_sampled_losses_77)))

for loss in list(set(selected_sampled_losses_77)):
    for img in label_77_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第78类的采样=================================================
selected_sampled_losses_78 = []
ms_78 = MeanShift()
ms_78.fit(label_78_loss)

labels_78 = ms_78.labels_  # 第0类中的每个样本对应的簇号

n_clusters_78 = len(np.unique(labels_78))  # 第0类样本有几个簇

need_sampled_cluster_78 = []  # 需要下采样的簇号

cluster_to_num_78 = Counter(labels_78)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_78.keys():
    if cluster_to_num_78[k] > len(labels_78) // n_clusters_78:
        need_sampled_cluster_78.append(k)

need_sampled_losses_78 = [[] for _ in range(len(need_sampled_cluster_78))]

for i in range(len(need_sampled_cluster_78)):
    for j in range(len(labels_78)):
        if labels_78[j] == need_sampled_cluster_78[i]:
            need_sampled_losses_78[i].append(label_78_loss[j][0])

for j in range(len(labels_78)):
    if labels_78[j] not in need_sampled_cluster_78:
        selected_sampled_losses_78.append(label_78_loss[j][0])

for loss in need_sampled_losses_78:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_78.append(lo)
print(len(label_78_loss))
print(len(set(selected_sampled_losses_78)))

for loss in list(set(selected_sampled_losses_78)):
    for img in label_78_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第79类的采样=================================================
selected_sampled_losses_79 = []
ms_79 = MeanShift()
ms_79.fit(label_79_loss)

labels_79 = ms_79.labels_  # 第0类中的每个样本对应的簇号

n_clusters_79 = len(np.unique(labels_79))  # 第0类样本有几个簇

need_sampled_cluster_79 = []  # 需要下采样的簇号

cluster_to_num_79 = Counter(labels_79)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_79.keys():
    if cluster_to_num_79[k] > len(labels_79) // n_clusters_79:
        need_sampled_cluster_79.append(k)

need_sampled_losses_79 = [[] for _ in range(len(need_sampled_cluster_79))]

for i in range(len(need_sampled_cluster_79)):
    for j in range(len(labels_79)):
        if labels_79[j] == need_sampled_cluster_79[i]:
            need_sampled_losses_79[i].append(label_79_loss[j][0])

for j in range(len(labels_79)):
    if labels_79[j] not in need_sampled_cluster_79:
        selected_sampled_losses_79.append(label_79_loss[j][0])

for loss in need_sampled_losses_79:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_79.append(lo)
print(len(label_79_loss))
print(len(set(selected_sampled_losses_79)))
for loss in list(set(selected_sampled_losses_79)):
    for img in label_79_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))


# ==================================================针对第80类的采样=================================================

selected_sampled_losses_80 = []

ms_80 = MeanShift()
ms_80.fit(label_80_loss)

labels_80 = ms_80.labels_  # 第0类中的每个样本对应的簇号

n_clusters_80 = len(np.unique(labels_80))  # 第0类样本有几个簇

need_sampled_cluster_80 = []  # 需要下采样的簇号

cluster_to_num_80 = Counter(labels_80)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_80.keys():
    if cluster_to_num_80[k] > len(labels_80) // n_clusters_80:
        need_sampled_cluster_80.append(k)

need_sampled_losses_80 = [[] for _ in range(len(need_sampled_cluster_80))]

for i in range(len(need_sampled_cluster_80)):
    for j in range(len(labels_80)):
        if labels_80[j] == need_sampled_cluster_80[i]:
            need_sampled_losses_80[i].append(label_80_loss[j][0])

for j in range(len(labels_80)):
    if labels_80[j] not in need_sampled_cluster_80:
        selected_sampled_losses_80.append(label_80_loss[j][0])

for loss in need_sampled_losses_80:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_80.append(lo)
print(len(label_80_loss))
print(len(set(selected_sampled_losses_80)))

for loss in list(set(selected_sampled_losses_80)):
    for img in label_80_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第81类的采样=================================================
selected_sampled_losses_81 = []

ms_81 = MeanShift()
ms_81.fit(label_81_loss)

labels_81 = ms_81.labels_  # 第0类中的每个样本对应的簇号

n_clusters_81 = len(np.unique(labels_81))  # 第0类样本有几个簇

need_sampled_cluster_81 = []  # 需要下采样的簇号

cluster_to_num_81 = Counter(labels_81)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_81.keys():
    if cluster_to_num_81[k] > len(labels_81) // n_clusters_81:
        need_sampled_cluster_81.append(k)

need_sampled_losses_81 = [[] for _ in range(len(need_sampled_cluster_81))]

for i in range(len(need_sampled_cluster_81)):
    for j in range(len(labels_81)):
        if labels_81[j] == need_sampled_cluster_81[i]:
            need_sampled_losses_81[i].append(label_81_loss[j][0])

for j in range(len(labels_81)):
    if labels_81[j] not in need_sampled_cluster_81:
        selected_sampled_losses_81.append(label_81_loss[j][0])

for loss in need_sampled_losses_81:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_81.append(lo)
print(len(label_81_loss))
print(len(set(selected_sampled_losses_81)))

for loss in list(set(selected_sampled_losses_81)):
    for img in label_81_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第82类的采样=================================================
selected_sampled_losses_82 = []

ms_82 = MeanShift()
ms_82.fit(label_82_loss)

labels_82 = ms_82.labels_  # 第0类中的每个样本对应的簇号

n_clusters_82 = len(np.unique(labels_82))  # 第0类样本有几个簇

need_sampled_cluster_82 = []  # 需要下采样的簇号

cluster_to_num_82 = Counter(labels_82)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_82.keys():
    if cluster_to_num_82[k] > len(labels_82) // n_clusters_82:
        need_sampled_cluster_82.append(k)

need_sampled_losses_82 = [[] for _ in range(len(need_sampled_cluster_82))]

for i in range(len(need_sampled_cluster_82)):
    for j in range(len(labels_82)):
        if labels_82[j] == need_sampled_cluster_82[i]:
            need_sampled_losses_82[i].append(label_82_loss[j][0])

for j in range(len(labels_82)):
    if labels_82[j] not in need_sampled_cluster_82:
        selected_sampled_losses_82.append(label_82_loss[j][0])

for loss in need_sampled_losses_82:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_82.append(lo)
print(len(label_82_loss))
print(len(set(selected_sampled_losses_82)))

for loss in list(set(selected_sampled_losses_82)):
    for img in label_82_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第83类的采样=================================================
selected_sampled_losses_83 = []

ms_83 = MeanShift()
ms_83.fit(label_83_loss)

labels_83 = ms_83.labels_  # 第0类中的每个样本对应的簇号

n_clusters_83 = len(np.unique(labels_83))  # 第0类样本有几个簇

need_sampled_cluster_83 = []  # 需要下采样的簇号

cluster_to_num_83 = Counter(labels_83)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_83.keys():
    if cluster_to_num_83[k] > len(labels_83) // n_clusters_83:
        need_sampled_cluster_83.append(k)

need_sampled_losses_83 = [[] for _ in range(len(need_sampled_cluster_83))]

for i in range(len(need_sampled_cluster_83)):
    for j in range(len(labels_83)):
        if labels_83[j] == need_sampled_cluster_83[i]:
            need_sampled_losses_83[i].append(label_83_loss[j][0])

for j in range(len(labels_83)):
    if labels_83[j] not in need_sampled_cluster_83:
        selected_sampled_losses_83.append(label_83_loss[j][0])

for loss in need_sampled_losses_83:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_83.append(lo)
print(len(label_83_loss))
print(len(set(selected_sampled_losses_83)))

for loss in list(set(selected_sampled_losses_83)):
    for img in label_83_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第84类的采样=================================================
selected_sampled_losses_84 = []
ms_84 = MeanShift()
ms_84.fit(label_84_loss)

labels_84 = ms_84.labels_  # 第0类中的每个样本对应的簇号

n_clusters_84 = len(np.unique(labels_84))  # 第0类样本有几个簇

need_sampled_cluster_84 = []  # 需要下采样的簇号

cluster_to_num_84 = Counter(labels_84)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_84.keys():
    if cluster_to_num_84[k] > len(labels_84) // n_clusters_84:
        need_sampled_cluster_84.append(k)

need_sampled_losses_84 = [[] for _ in range(len(need_sampled_cluster_84))]

for i in range(len(need_sampled_cluster_84)):
    for j in range(len(labels_84)):
        if labels_84[j] == need_sampled_cluster_84[i]:
            need_sampled_losses_84[i].append(label_84_loss[j][0])

for j in range(len(labels_84)):
    if labels_84[j] not in need_sampled_cluster_84:
        selected_sampled_losses_84.append(label_84_loss[j][0])

for loss in need_sampled_losses_84:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_84.append(lo)
print(len(label_84_loss))
print(len(set(selected_sampled_losses_84)))

for loss in list(set(selected_sampled_losses_84)):
    for img in label_84_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第85类的采样=================================================
selected_sampled_losses_85 = []
ms_85 = MeanShift()
ms_85.fit(label_85_loss)

labels_85 = ms_85.labels_  # 第0类中的每个样本对应的簇号

n_clusters_85 = len(np.unique(labels_85))  # 第0类样本有几个簇

need_sampled_cluster_85 = []  # 需要下采样的簇号

cluster_to_num_85 = Counter(labels_85)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_85.keys():
    if cluster_to_num_85[k] > len(labels_85) // n_clusters_85:
        need_sampled_cluster_85.append(k)

need_sampled_losses_85 = [[] for _ in range(len(need_sampled_cluster_85))]

for i in range(len(need_sampled_cluster_85)):
    for j in range(len(labels_85)):
        if labels_85[j] == need_sampled_cluster_85[i]:
            need_sampled_losses_85[i].append(label_85_loss[j][0])

for j in range(len(labels_85)):
    if labels_85[j] not in need_sampled_cluster_85:
        selected_sampled_losses_85.append(label_85_loss[j][0])

for loss in need_sampled_losses_85:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_85.append(lo)
print(len(label_85_loss))
print(len(set(selected_sampled_losses_85)))

for loss in list(set(selected_sampled_losses_85)):
    for img in label_85_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第86类的采样=================================================
selected_sampled_losses_86 = []
ms_86 = MeanShift()
ms_86.fit(label_86_loss)

labels_86 = ms_86.labels_  # 第0类中的每个样本对应的簇号

n_clusters_86 = len(np.unique(labels_86))  # 第0类样本有几个簇

need_sampled_cluster_86 = []  # 需要下采样的簇号

cluster_to_num_86 = Counter(labels_86)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_86.keys():
    if cluster_to_num_86[k] > len(labels_86) // n_clusters_86:
        need_sampled_cluster_86.append(k)

need_sampled_losses_86 = [[] for _ in range(len(need_sampled_cluster_86))]

for i in range(len(need_sampled_cluster_86)):
    for j in range(len(labels_86)):
        if labels_86[j] == need_sampled_cluster_86[i]:
            need_sampled_losses_86[i].append(label_86_loss[j][0])

for j in range(len(labels_86)):
    if labels_86[j] not in need_sampled_cluster_86:
        selected_sampled_losses_86.append(label_86_loss[j][0])

for loss in need_sampled_losses_86:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_86.append(lo)
print(len(label_86_loss))
print(len(set(selected_sampled_losses_86)))

for loss in list(set(selected_sampled_losses_86)):
    for img in label_86_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第87类的采样=================================================
selected_sampled_losses_87 = []
ms_87 = MeanShift()
ms_87.fit(label_87_loss)

labels_87 = ms_87.labels_  # 第0类中的每个样本对应的簇号

n_clusters_87 = len(np.unique(labels_87))  # 第0类样本有几个簇

need_sampled_cluster_87 = []  # 需要下采样的簇号

cluster_to_num_87 = Counter(labels_87)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_87.keys():
    if cluster_to_num_87[k] > len(labels_87) // n_clusters_87:
        need_sampled_cluster_87.append(k)

need_sampled_losses_87 = [[] for _ in range(len(need_sampled_cluster_87))]

for i in range(len(need_sampled_cluster_87)):
    for j in range(len(labels_87)):
        if labels_87[j] == need_sampled_cluster_87[i]:
            need_sampled_losses_87[i].append(label_87_loss[j][0])

for j in range(len(labels_87)):
    if labels_87[j] not in need_sampled_cluster_87:
        selected_sampled_losses_87.append(label_87_loss[j][0])

for loss in need_sampled_losses_87:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_87.append(lo)
print(len(label_87_loss))
print(len(set(selected_sampled_losses_87)))

for loss in list(set(selected_sampled_losses_87)):
    for img in label_87_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第88类的采样=================================================
selected_sampled_losses_88 = []
ms_88 = MeanShift()
ms_88.fit(label_88_loss)

labels_88 = ms_88.labels_  # 第0类中的每个样本对应的簇号

n_clusters_88 = len(np.unique(labels_88))  # 第0类样本有几个簇

need_sampled_cluster_88 = []  # 需要下采样的簇号

cluster_to_num_88 = Counter(labels_88)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_88.keys():
    if cluster_to_num_88[k] > len(labels_88) // n_clusters_88:
        need_sampled_cluster_88.append(k)

need_sampled_losses_88 = [[] for _ in range(len(need_sampled_cluster_88))]

for i in range(len(need_sampled_cluster_88)):
    for j in range(len(labels_88)):
        if labels_88[j] == need_sampled_cluster_88[i]:
            need_sampled_losses_88[i].append(label_88_loss[j][0])

for j in range(len(labels_88)):
    if labels_88[j] not in need_sampled_cluster_88:
        selected_sampled_losses_88.append(label_88_loss[j][0])

for loss in need_sampled_losses_88:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_88.append(lo)
print(len(label_88_loss))
print(len(set(selected_sampled_losses_88)))

for loss in list(set(selected_sampled_losses_88)):
    for img in label_88_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第89类的采样=================================================
selected_sampled_losses_89 = []
ms_89 = MeanShift()
ms_89.fit(label_89_loss)

labels_89 = ms_89.labels_  # 第0类中的每个样本对应的簇号

n_clusters_89 = len(np.unique(labels_89))  # 第0类样本有几个簇

need_sampled_cluster_89 = []  # 需要下采样的簇号

cluster_to_num_89 = Counter(labels_89)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_89.keys():
    if cluster_to_num_89[k] > len(labels_89) // n_clusters_89:
        need_sampled_cluster_89.append(k)

need_sampled_losses_89 = [[] for _ in range(len(need_sampled_cluster_89))]

for i in range(len(need_sampled_cluster_89)):
    for j in range(len(labels_89)):
        if labels_89[j] == need_sampled_cluster_89[i]:
            need_sampled_losses_89[i].append(label_89_loss[j][0])

for j in range(len(labels_89)):
    if labels_89[j] not in need_sampled_cluster_89:
        selected_sampled_losses_89.append(label_89_loss[j][0])

for loss in need_sampled_losses_89:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_89.append(lo)
print(len(label_89_loss))
print(len(set(selected_sampled_losses_89)))

for loss in list(set(selected_sampled_losses_89)):
    for img in label_89_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第90类的采样=================================================
selected_sampled_losses_90 = []
ms_90 = MeanShift()
ms_90.fit(label_90_loss)

labels_90 = ms_90.labels_  # 第0类中的每个样本对应的簇号

n_clusters_90 = len(np.unique(labels_90))  # 第0类样本有几个簇

need_sampled_cluster_90 = []  # 需要下采样的簇号

cluster_to_num_90 = Counter(labels_90)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_90.keys():
    if cluster_to_num_90[k] > len(labels_90) // n_clusters_90:
        need_sampled_cluster_90.append(k)

need_sampled_losses_90 = [[] for _ in range(len(need_sampled_cluster_90))]

for i in range(len(need_sampled_cluster_90)):
    for j in range(len(labels_90)):
        if labels_90[j] == need_sampled_cluster_90[i]:
            need_sampled_losses_90[i].append(label_90_loss[j][0])

for j in range(len(labels_90)):
    if labels_90[j] not in need_sampled_cluster_90:
        selected_sampled_losses_90.append(label_90_loss[j][0])

for loss in need_sampled_losses_90:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_90.append(lo)
print(len(label_90_loss))
print(len(set(selected_sampled_losses_90)))

for loss in list(set(selected_sampled_losses_90)):
    for img in label_90_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第91类的采样=================================================
selected_sampled_losses_91 = []

ms_91 = MeanShift()
ms_91.fit(label_91_loss)

labels_91 = ms_91.labels_  # 第0类中的每个样本对应的簇号

n_clusters_91 = len(np.unique(labels_91))  # 第0类样本有几个簇

need_sampled_cluster_91 = []  # 需要下采样的簇号

cluster_to_num_91 = Counter(labels_91)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_91.keys():
    if cluster_to_num_91[k] > len(labels_91) // n_clusters_91:
        need_sampled_cluster_91.append(k)

need_sampled_losses_91 = [[] for _ in range(len(need_sampled_cluster_91))]

for i in range(len(need_sampled_cluster_91)):
    for j in range(len(labels_91)):
        if labels_91[j] == need_sampled_cluster_91[i]:
            need_sampled_losses_91[i].append(label_91_loss[j][0])

for j in range(len(labels_91)):
    if labels_91[j] not in need_sampled_cluster_91:
        selected_sampled_losses_91.append(label_91_loss[j][0])

for loss in need_sampled_losses_91:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_91.append(lo)
print(len(label_91_loss))
print(len(set(selected_sampled_losses_91)))

for loss in list(set(selected_sampled_losses_91)):
    for img in label_91_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第92类的采样=================================================
selected_sampled_losses_92 = []
ms_92 = MeanShift()
ms_92.fit(label_92_loss)

labels_92 = ms_92.labels_  # 第0类中的每个样本对应的簇号

n_clusters_92 = len(np.unique(labels_92))  # 第0类样本有几个簇

need_sampled_cluster_92 = []  # 需要下采样的簇号

cluster_to_num_92 = Counter(labels_92)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_92.keys():
    if cluster_to_num_92[k] > len(labels_92) // n_clusters_92:
        need_sampled_cluster_92.append(k)

need_sampled_losses_92 = [[] for _ in range(len(need_sampled_cluster_92))]

for i in range(len(need_sampled_cluster_92)):
    for j in range(len(labels_92)):
        if labels_92[j] == need_sampled_cluster_92[i]:
            need_sampled_losses_92[i].append(label_92_loss[j][0])

for j in range(len(labels_92)):
    if labels_92[j] not in need_sampled_cluster_92:
        selected_sampled_losses_92.append(label_92_loss[j][0])

for loss in need_sampled_losses_92:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_92.append(lo)
print(len(label_92_loss))
print(len(set(selected_sampled_losses_92)))

for loss in list(set(selected_sampled_losses_92)):
    for img in label_92_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第93类的采样=================================================
selected_sampled_losses_93 = []
ms_93 = MeanShift()
ms_93.fit(label_93_loss)

labels_93 = ms_93.labels_  # 第0类中的每个样本对应的簇号

n_clusters_93 = len(np.unique(labels_93))  # 第0类样本有几个簇

need_sampled_cluster_93 = []  # 需要下采样的簇号

cluster_to_num_93 = Counter(labels_93)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_93.keys():
    if cluster_to_num_93[k] > len(labels_93) // n_clusters_93:
        need_sampled_cluster_93.append(k)

need_sampled_losses_93 = [[] for _ in range(len(need_sampled_cluster_93))]

for i in range(len(need_sampled_cluster_93)):
    for j in range(len(labels_93)):
        if labels_93[j] == need_sampled_cluster_93[i]:
            need_sampled_losses_93[i].append(label_93_loss[j][0])

for j in range(len(labels_93)):
    if labels_93[j] not in need_sampled_cluster_93:
        selected_sampled_losses_93.append(label_93_loss[j][0])

for loss in need_sampled_losses_93:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_93.append(lo)
print(len(label_93_loss))
print(len(set(selected_sampled_losses_93)))

for loss in list(set(selected_sampled_losses_93)):
    for img in label_93_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第94类的采样=================================================
selected_sampled_losses_94 = []
ms_94 = MeanShift()
ms_94.fit(label_94_loss)

labels_94 = ms_94.labels_  # 第0类中的每个样本对应的簇号

n_clusters_94 = len(np.unique(labels_94))  # 第0类样本有几个簇

need_sampled_cluster_94 = []  # 需要下采样的簇号

cluster_to_num_94 = Counter(labels_94)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_94.keys():
    if cluster_to_num_94[k] > len(labels_94) // n_clusters_94:
        need_sampled_cluster_94.append(k)

need_sampled_losses_94 = [[] for _ in range(len(need_sampled_cluster_94))]

for i in range(len(need_sampled_cluster_94)):
    for j in range(len(labels_94)):
        if labels_94[j] == need_sampled_cluster_94[i]:
            need_sampled_losses_94[i].append(label_94_loss[j][0])

for j in range(len(labels_94)):
    if labels_94[j] not in need_sampled_cluster_94:
        selected_sampled_losses_94.append(label_94_loss[j][0])

for loss in need_sampled_losses_94:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_94.append(lo)
print(len(label_94_loss))
print(len(set(selected_sampled_losses_94)))

for loss in list(set(selected_sampled_losses_94)):
    for img in label_94_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第95类的采样=================================================
selected_sampled_losses_95 = []
ms_95 = MeanShift()
ms_95.fit(label_95_loss)

labels_95 = ms_95.labels_  # 第0类中的每个样本对应的簇号

n_clusters_95 = len(np.unique(labels_95))  # 第0类样本有几个簇

need_sampled_cluster_95 = []  # 需要下采样的簇号

cluster_to_num_95 = Counter(labels_95)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_95.keys():
    if cluster_to_num_95[k] > len(labels_95) // n_clusters_95:
        need_sampled_cluster_95.append(k)

need_sampled_losses_95 = [[] for _ in range(len(need_sampled_cluster_95))]

for i in range(len(need_sampled_cluster_95)):
    for j in range(len(labels_95)):
        if labels_95[j] == need_sampled_cluster_95[i]:
            need_sampled_losses_95[i].append(label_95_loss[j][0])

for j in range(len(labels_95)):
    if labels_95[j] not in need_sampled_cluster_95:
        selected_sampled_losses_95.append(label_95_loss[j][0])

for loss in need_sampled_losses_95:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_95.append(lo)
print(len(label_95_loss))
print(len(set(selected_sampled_losses_95)))

for loss in list(set(selected_sampled_losses_95)):
    for img in label_95_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第96类的采样=================================================
selected_sampled_losses_96 = []
ms_96 = MeanShift()
ms_96.fit(label_96_loss)

labels_96 = ms_96.labels_  # 第0类中的每个样本对应的簇号

n_clusters_96 = len(np.unique(labels_96))  # 第0类样本有几个簇

need_sampled_cluster_96 = []  # 需要下采样的簇号

cluster_to_num_96 = Counter(labels_96)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_96.keys():
    if cluster_to_num_96[k] > len(labels_96) // n_clusters_96:
        need_sampled_cluster_96.append(k)

need_sampled_losses_96 = [[] for _ in range(len(need_sampled_cluster_96))]

for i in range(len(need_sampled_cluster_96)):
    for j in range(len(labels_96)):
        if labels_96[j] == need_sampled_cluster_96[i]:
            need_sampled_losses_96[i].append(label_96_loss[j][0])

for j in range(len(labels_96)):
    if labels_96[j] not in need_sampled_cluster_96:
        selected_sampled_losses_96.append(label_96_loss[j][0])

for loss in need_sampled_losses_96:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_96.append(lo)
print(len(label_96_loss))
print(len(set(selected_sampled_losses_96)))

for loss in list(set(selected_sampled_losses_96)):
    for img in label_96_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第97类的采样=================================================
selected_sampled_losses_97 = []
ms_97 = MeanShift()
ms_97.fit(label_97_loss)

labels_97 = ms_97.labels_  # 第0类中的每个样本对应的簇号

n_clusters_97 = len(np.unique(labels_97))  # 第0类样本有几个簇

need_sampled_cluster_97 = []  # 需要下采样的簇号

cluster_to_num_97 = Counter(labels_97)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_97.keys():
    if cluster_to_num_97[k] > len(labels_97) // n_clusters_97:
        need_sampled_cluster_97.append(k)

need_sampled_losses_97 = [[] for _ in range(len(need_sampled_cluster_97))]

for i in range(len(need_sampled_cluster_97)):
    for j in range(len(labels_97)):
        if labels_97[j] == need_sampled_cluster_97[i]:
            need_sampled_losses_97[i].append(label_97_loss[j][0])

for j in range(len(labels_97)):
    if labels_97[j] not in need_sampled_cluster_97:
        selected_sampled_losses_97.append(label_97_loss[j][0])

for loss in need_sampled_losses_97:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_97.append(lo)
print(len(label_97_loss))
print(len(set(selected_sampled_losses_97)))

for loss in list(set(selected_sampled_losses_97)):
    for img in label_97_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第98类的采样=================================================
selected_sampled_losses_98 = []
ms_98 = MeanShift()
ms_98.fit(label_98_loss)

labels_98 = ms_98.labels_  # 第0类中的每个样本对应的簇号

n_clusters_98 = len(np.unique(labels_98))  # 第0类样本有几个簇

need_sampled_cluster_98 = []  # 需要下采样的簇号

cluster_to_num_98 = Counter(labels_98)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_98.keys():
    if cluster_to_num_98[k] > len(labels_98) // n_clusters_98:
        need_sampled_cluster_98.append(k)

need_sampled_losses_98 = [[] for _ in range(len(need_sampled_cluster_98))]

for i in range(len(need_sampled_cluster_98)):
    for j in range(len(labels_98)):
        if labels_98[j] == need_sampled_cluster_98[i]:
            need_sampled_losses_98[i].append(label_98_loss[j][0])

for j in range(len(labels_98)):
    if labels_98[j] not in need_sampled_cluster_98:
        selected_sampled_losses_98.append(label_98_loss[j][0])

for loss in need_sampled_losses_98:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_98.append(lo)
print(len(label_98_loss))
print(len(set(selected_sampled_losses_98)))

for loss in list(set(selected_sampled_losses_98)):
    for img in label_98_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第99类的采样=================================================
selected_sampled_losses_99 = []
ms_99 = MeanShift()
ms_99.fit(label_99_loss)

labels_99 = ms_99.labels_  # 第0类中的每个样本对应的簇号

n_clusters_99 = len(np.unique(labels_99))  # 第0类样本有几个簇

need_sampled_cluster_99 = []  # 需要下采样的簇号

cluster_to_num_99 = Counter(labels_99)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_99.keys():
    if cluster_to_num_99[k] > len(labels_99) // n_clusters_99:
        need_sampled_cluster_99.append(k)

need_sampled_losses_99 = [[] for _ in range(len(need_sampled_cluster_99))]

for i in range(len(need_sampled_cluster_99)):
    for j in range(len(labels_99)):
        if labels_99[j] == need_sampled_cluster_99[i]:
            need_sampled_losses_99[i].append(label_99_loss[j][0])

for j in range(len(labels_99)):
    if labels_99[j] not in need_sampled_cluster_99:
        selected_sampled_losses_99.append(label_99_loss[j][0])

for loss in need_sampled_losses_99:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_99.append(lo)
print(len(label_99_loss))
print(len(set(selected_sampled_losses_99)))
for loss in list(set(selected_sampled_losses_99)):
    for img in label_99_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))

# ==================================================针对第100类的采样=================================================

selected_sampled_losses_100 = []

ms_100 = MeanShift()
ms_100.fit(label_100_loss)

labels_100 = ms_100.labels_  # 第_100类中的每个样本对应的簇号

n_clusters_100 = len(np.unique(labels_100))  # 第_100类样本有几个簇

need_sampled_cluster_100 = []  # 需要下采样的簇号

cluster_to_num_100 = Counter(labels_100)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_100.keys():
    if cluster_to_num_100[k] > len(labels_100) // n_clusters_100:
        need_sampled_cluster_100.append(k)

need_sampled_losses_100 = [[] for _ in range(len(need_sampled_cluster_100))]

for i in range(len(need_sampled_cluster_100)):
    for j in range(len(labels_100)):
        if labels_100[j] == need_sampled_cluster_100[i]:
            need_sampled_losses_100[i].append(label_100_loss[j][0])

for j in range(len(labels_100)):
    if labels_100[j] not in need_sampled_cluster_100:
        selected_sampled_losses_100.append(label_100_loss[j][0])

for loss in need_sampled_losses_100:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_100.append(lo)
print(len(label_100_loss))
print(len(set(selected_sampled_losses_100)))

for loss in list(set(selected_sampled_losses_100)):
    for img in label_100_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第101类的采样=================================================
selected_sampled_losses_101 = []

ms_101 = MeanShift()
ms_101.fit(label_101_loss)

labels_101 = ms_101.labels_  # 第0类中的每个样本对应的簇号

n_clusters_101 = len(np.unique(labels_101))  # 第0类样本有几个簇

need_sampled_cluster_101 = []  # 需要下采样的簇号

cluster_to_num_101 = Counter(labels_101)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_101.keys():
    if cluster_to_num_101[k] > len(labels_101) // n_clusters_101:
        need_sampled_cluster_101.append(k)

need_sampled_losses_101 = [[] for _ in range(len(need_sampled_cluster_101))]

for i in range(len(need_sampled_cluster_101)):
    for j in range(len(labels_101)):
        if labels_101[j] == need_sampled_cluster_101[i]:
            need_sampled_losses_101[i].append(label_101_loss[j][0])

for j in range(len(labels_101)):
    if labels_101[j] not in need_sampled_cluster_101:
        selected_sampled_losses_101.append(label_101_loss[j][0])

for loss in need_sampled_losses_101:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_101.append(lo)
print(len(label_101_loss))
print(len(set(selected_sampled_losses_101)))

for loss in list(set(selected_sampled_losses_101)):
    for img in label_101_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第102类的采样=================================================
selected_sampled_losses_102 = []

ms_102 = MeanShift()
ms_102.fit(label_102_loss)

labels_102 = ms_102.labels_  # 第0类中的每个样本对应的簇号

n_clusters_102 = len(np.unique(labels_102))  # 第0类样本有几个簇

need_sampled_cluster_102 = []  # 需要下采样的簇号

cluster_to_num_102 = Counter(labels_102)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_102.keys():
    if cluster_to_num_102[k] > len(labels_102) // n_clusters_102:
        need_sampled_cluster_102.append(k)

need_sampled_losses_102 = [[] for _ in range(len(need_sampled_cluster_102))]

for i in range(len(need_sampled_cluster_102)):
    for j in range(len(labels_102)):
        if labels_102[j] == need_sampled_cluster_102[i]:
            need_sampled_losses_102[i].append(label_102_loss[j][0])

for j in range(len(labels_102)):
    if labels_102[j] not in need_sampled_cluster_102:
        selected_sampled_losses_102.append(label_102_loss[j][0])

for loss in need_sampled_losses_102:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_102.append(lo)
print(len(label_102_loss))
print(len(set(selected_sampled_losses_102)))

for loss in list(set(selected_sampled_losses_102)):
    for img in label_102_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第103类的采样=================================================
selected_sampled_losses_103 = []

ms_103 = MeanShift()
ms_103.fit(label_103_loss)

labels_103 = ms_103.labels_  # 第0类中的每个样本对应的簇号

n_clusters_103 = len(np.unique(labels_103))  # 第0类样本有几个簇

need_sampled_cluster_103 = []  # 需要下采样的簇号

cluster_to_num_103 = Counter(labels_103)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_103.keys():
    if cluster_to_num_103[k] > len(labels_103) // n_clusters_103:
        need_sampled_cluster_103.append(k)

need_sampled_losses_103 = [[] for _ in range(len(need_sampled_cluster_103))]

for i in range(len(need_sampled_cluster_103)):
    for j in range(len(labels_103)):
        if labels_103[j] == need_sampled_cluster_103[i]:
            need_sampled_losses_103[i].append(label_103_loss[j][0])

for j in range(len(labels_103)):
    if labels_103[j] not in need_sampled_cluster_103:
        selected_sampled_losses_103.append(label_103_loss[j][0])

for loss in need_sampled_losses_103:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_103.append(lo)
print(len(label_103_loss))
print(len(set(selected_sampled_losses_103)))

for loss in list(set(selected_sampled_losses_103)):
    for img in label_103_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第104类的采样=================================================
selected_sampled_losses_104 = []
ms_104 = MeanShift()
ms_104.fit(label_104_loss)

labels_104 = ms_104.labels_  # 第0类中的每个样本对应的簇号

n_clusters_104 = len(np.unique(labels_104))  # 第0类样本有几个簇

need_sampled_cluster_104 = []  # 需要下采样的簇号

cluster_to_num_104 = Counter(labels_104)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_104.keys():
    if cluster_to_num_104[k] > len(labels_104) // n_clusters_104:
        need_sampled_cluster_104.append(k)

need_sampled_losses_104 = [[] for _ in range(len(need_sampled_cluster_104))]

for i in range(len(need_sampled_cluster_104)):
    for j in range(len(labels_104)):
        if labels_104[j] == need_sampled_cluster_104[i]:
            need_sampled_losses_104[i].append(label_104_loss[j][0])

for j in range(len(labels_104)):
    if labels_104[j] not in need_sampled_cluster_104:
        selected_sampled_losses_104.append(label_104_loss[j][0])

for loss in need_sampled_losses_104:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_104.append(lo)
print(len(label_104_loss))
print(len(set(selected_sampled_losses_104)))

for loss in list(set(selected_sampled_losses_104)):
    for img in label_104_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第105类的采样=================================================
selected_sampled_losses_105 = []
ms_105 = MeanShift()
ms_105.fit(label_105_loss)

labels_105 = ms_105.labels_  # 第0类中的每个样本对应的簇号

n_clusters_105 = len(np.unique(labels_105))  # 第0类样本有几个簇

need_sampled_cluster_105 = []  # 需要下采样的簇号

cluster_to_num_105 = Counter(labels_105)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_105.keys():
    if cluster_to_num_105[k] > len(labels_105) // n_clusters_105:
        need_sampled_cluster_105.append(k)

need_sampled_losses_105 = [[] for _ in range(len(need_sampled_cluster_105))]

for i in range(len(need_sampled_cluster_105)):
    for j in range(len(labels_105)):
        if labels_105[j] == need_sampled_cluster_105[i]:
            need_sampled_losses_105[i].append(label_105_loss[j][0])

for j in range(len(labels_105)):
    if labels_105[j] not in need_sampled_cluster_105:
        selected_sampled_losses_105.append(label_105_loss[j][0])

for loss in need_sampled_losses_105:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_105.append(lo)
print(len(label_105_loss))
print(len(set(selected_sampled_losses_105)))

for loss in list(set(selected_sampled_losses_105)):
    for img in label_105_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第106类的采样=================================================
selected_sampled_losses_106 = []
ms_106 = MeanShift()
ms_106.fit(label_106_loss)

labels_106 = ms_106.labels_  # 第0类中的每个样本对应的簇号

n_clusters_106 = len(np.unique(labels_106))  # 第0类样本有几个簇

need_sampled_cluster_106 = []  # 需要下采样的簇号

cluster_to_num_106 = Counter(labels_106)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_106.keys():
    if cluster_to_num_106[k] > len(labels_106) // n_clusters_106:
        need_sampled_cluster_106.append(k)

need_sampled_losses_106 = [[] for _ in range(len(need_sampled_cluster_106))]

for i in range(len(need_sampled_cluster_106)):
    for j in range(len(labels_106)):
        if labels_106[j] == need_sampled_cluster_106[i]:
            need_sampled_losses_106[i].append(label_106_loss[j][0])

for j in range(len(labels_106)):
    if labels_106[j] not in need_sampled_cluster_106:
        selected_sampled_losses_106.append(label_106_loss[j][0])

for loss in need_sampled_losses_106:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_106.append(lo)
print(len(label_106_loss))
print(len(set(selected_sampled_losses_106)))

for loss in list(set(selected_sampled_losses_106)):
    for img in label_106_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第107类的采样=================================================
selected_sampled_losses_107 = []
ms_107 = MeanShift()
ms_107.fit(label_107_loss)

labels_107 = ms_107.labels_  # 第0类中的每个样本对应的簇号

n_clusters_107 = len(np.unique(labels_107))  # 第0类样本有几个簇

need_sampled_cluster_107 = []  # 需要下采样的簇号

cluster_to_num_107 = Counter(labels_107)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_107.keys():
    if cluster_to_num_107[k] > len(labels_107) // n_clusters_107:
        need_sampled_cluster_107.append(k)

need_sampled_losses_107 = [[] for _ in range(len(need_sampled_cluster_107))]

for i in range(len(need_sampled_cluster_107)):
    for j in range(len(labels_107)):
        if labels_107[j] == need_sampled_cluster_107[i]:
            need_sampled_losses_107[i].append(label_107_loss[j][0])

for j in range(len(labels_107)):
    if labels_107[j] not in need_sampled_cluster_107:
        selected_sampled_losses_107.append(label_107_loss[j][0])

for loss in need_sampled_losses_107:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_107.append(lo)
print(len(label_107_loss))
print(len(set(selected_sampled_losses_107)))

for loss in list(set(selected_sampled_losses_107)):
    for img in label_107_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第108类的采样=================================================
selected_sampled_losses_108 = []
ms_108 = MeanShift()
ms_108.fit(label_108_loss)

labels_108 = ms_108.labels_  # 第0类中的每个样本对应的簇号

n_clusters_108 = len(np.unique(labels_108))  # 第0类样本有几个簇

need_sampled_cluster_108 = []  # 需要下采样的簇号

cluster_to_num_108 = Counter(labels_108)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_108.keys():
    if cluster_to_num_108[k] > len(labels_108) // n_clusters_108:
        need_sampled_cluster_108.append(k)

need_sampled_losses_108 = [[] for _ in range(len(need_sampled_cluster_108))]

for i in range(len(need_sampled_cluster_108)):
    for j in range(len(labels_108)):
        if labels_108[j] == need_sampled_cluster_108[i]:
            need_sampled_losses_108[i].append(label_108_loss[j][0])

for j in range(len(labels_108)):
    if labels_108[j] not in need_sampled_cluster_108:
        selected_sampled_losses_108.append(label_108_loss[j][0])

for loss in need_sampled_losses_108:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_108.append(lo)
print(len(label_108_loss))
print(len(set(selected_sampled_losses_108)))

for loss in list(set(selected_sampled_losses_108)):
    for img in label_108_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第109类的采样=================================================
selected_sampled_losses_109 = []
ms_109 = MeanShift()
ms_109.fit(label_109_loss)

labels_109 = ms_109.labels_  # 第0类中的每个样本对应的簇号

n_clusters_109 = len(np.unique(labels_109))  # 第0类样本有几个簇

need_sampled_cluster_109 = []  # 需要下采样的簇号

cluster_to_num_109 = Counter(labels_109)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_109.keys():
    if cluster_to_num_109[k] > len(labels_109) // n_clusters_109:
        need_sampled_cluster_109.append(k)

need_sampled_losses_109 = [[] for _ in range(len(need_sampled_cluster_109))]

for i in range(len(need_sampled_cluster_109)):
    for j in range(len(labels_109)):
        if labels_109[j] == need_sampled_cluster_109[i]:
            need_sampled_losses_109[i].append(label_109_loss[j][0])

for j in range(len(labels_109)):
    if labels_109[j] not in need_sampled_cluster_109:
        selected_sampled_losses_109.append(label_109_loss[j][0])

for loss in need_sampled_losses_109:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_109.append(lo)
print(len(label_109_loss))
print(len(set(selected_sampled_losses_109)))

for loss in list(set(selected_sampled_losses_109)):
    for img in label_109_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第110类的采样=================================================
selected_sampled_losses_110 = []
ms_110 = MeanShift()
ms_110.fit(label_110_loss)

labels_110 = ms_110.labels_  # 第0类中的每个样本对应的簇号

n_clusters_110 = len(np.unique(labels_110))  # 第0类样本有几个簇

need_sampled_cluster_110 = []  # 需要下采样的簇号

cluster_to_num_110 = Counter(labels_110)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_110.keys():
    if cluster_to_num_110[k] > len(labels_110) // n_clusters_110:
        need_sampled_cluster_110.append(k)

need_sampled_losses_110 = [[] for _ in range(len(need_sampled_cluster_110))]

for i in range(len(need_sampled_cluster_110)):
    for j in range(len(labels_110)):
        if labels_110[j] == need_sampled_cluster_110[i]:
            need_sampled_losses_110[i].append(label_110_loss[j][0])

for j in range(len(labels_110)):
    if labels_110[j] not in need_sampled_cluster_110:
        selected_sampled_losses_110.append(label_110_loss[j][0])

for loss in need_sampled_losses_110:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_110.append(lo)
print(len(label_110_loss))
print(len(set(selected_sampled_losses_110)))

for loss in list(set(selected_sampled_losses_110)):
    for img in label_110_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第111类的采样=================================================
selected_sampled_losses_111 = []

ms_111 = MeanShift()
ms_111.fit(label_111_loss)

labels_111 = ms_111.labels_  # 第0类中的每个样本对应的簇号

n_clusters_111 = len(np.unique(labels_111))  # 第0类样本有几个簇

need_sampled_cluster_111 = []  # 需要下采样的簇号

cluster_to_num_111 = Counter(labels_111)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_111.keys():
    if cluster_to_num_111[k] > len(labels_111) // n_clusters_111:
        need_sampled_cluster_111.append(k)

need_sampled_losses_111 = [[] for _ in range(len(need_sampled_cluster_111))]

for i in range(len(need_sampled_cluster_111)):
    for j in range(len(labels_111)):
        if labels_111[j] == need_sampled_cluster_111[i]:
            need_sampled_losses_111[i].append(label_111_loss[j][0])

for j in range(len(labels_111)):
    if labels_111[j] not in need_sampled_cluster_111:
        selected_sampled_losses_111.append(label_111_loss[j][0])

for loss in need_sampled_losses_111:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_111.append(lo)
print(len(label_111_loss))
print(len(set(selected_sampled_losses_111)))

for loss in list(set(selected_sampled_losses_111)):
    for img in label_111_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第112类的采样=================================================
selected_sampled_losses_112 = []
ms_112 = MeanShift()
ms_112.fit(label_112_loss)

labels_112 = ms_112.labels_  # 第0类中的每个样本对应的簇号

n_clusters_112 = len(np.unique(labels_112))  # 第0类样本有几个簇

need_sampled_cluster_112 = []  # 需要下采样的簇号

cluster_to_num_112 = Counter(labels_112)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_112.keys():
    if cluster_to_num_112[k] > len(labels_112) // n_clusters_112:
        need_sampled_cluster_112.append(k)

need_sampled_losses_112 = [[] for _ in range(len(need_sampled_cluster_112))]

for i in range(len(need_sampled_cluster_112)):
    for j in range(len(labels_112)):
        if labels_112[j] == need_sampled_cluster_112[i]:
            need_sampled_losses_112[i].append(label_112_loss[j][0])

for j in range(len(labels_112)):
    if labels_112[j] not in need_sampled_cluster_112:
        selected_sampled_losses_112.append(label_112_loss[j][0])

for loss in need_sampled_losses_112:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_112.append(lo)
print(len(label_112_loss))
print(len(set(selected_sampled_losses_112)))

for loss in list(set(selected_sampled_losses_112)):
    for img in label_112_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第113类的采样=================================================
selected_sampled_losses_113 = []
ms_113 = MeanShift()
ms_113.fit(label_113_loss)

labels_113 = ms_113.labels_  # 第0类中的每个样本对应的簇号

n_clusters_113 = len(np.unique(labels_113))  # 第0类样本有几个簇

need_sampled_cluster_113 = []  # 需要下采样的簇号

cluster_to_num_113 = Counter(labels_113)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_113.keys():
    if cluster_to_num_113[k] > len(labels_113) // n_clusters_113:
        need_sampled_cluster_113.append(k)

need_sampled_losses_113 = [[] for _ in range(len(need_sampled_cluster_113))]

for i in range(len(need_sampled_cluster_113)):
    for j in range(len(labels_113)):
        if labels_113[j] == need_sampled_cluster_113[i]:
            need_sampled_losses_113[i].append(label_113_loss[j][0])

for j in range(len(labels_113)):
    if labels_113[j] not in need_sampled_cluster_113:
        selected_sampled_losses_113.append(label_113_loss[j][0])

for loss in need_sampled_losses_113:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_113.append(lo)
print(len(label_113_loss))
print(len(set(selected_sampled_losses_113)))

for loss in list(set(selected_sampled_losses_113)):
    for img in label_113_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第114类的采样=================================================
selected_sampled_losses_114 = []
ms_114 = MeanShift()
ms_114.fit(label_114_loss)

labels_114 = ms_114.labels_  # 第0类中的每个样本对应的簇号

n_clusters_114 = len(np.unique(labels_114))  # 第0类样本有几个簇

need_sampled_cluster_114 = []  # 需要下采样的簇号

cluster_to_num_114 = Counter(labels_114)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_114.keys():
    if cluster_to_num_114[k] > len(labels_114) // n_clusters_114:
        need_sampled_cluster_114.append(k)

need_sampled_losses_114 = [[] for _ in range(len(need_sampled_cluster_114))]

for i in range(len(need_sampled_cluster_114)):
    for j in range(len(labels_114)):
        if labels_114[j] == need_sampled_cluster_114[i]:
            need_sampled_losses_114[i].append(label_114_loss[j][0])

for j in range(len(labels_114)):
    if labels_114[j] not in need_sampled_cluster_114:
        selected_sampled_losses_114.append(label_114_loss[j][0])

for loss in need_sampled_losses_114:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_114.append(lo)
print(len(label_114_loss))
print(len(set(selected_sampled_losses_114)))

for loss in list(set(selected_sampled_losses_114)):
    for img in label_114_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第115类的采样=================================================
selected_sampled_losses_115 = []
ms_115 = MeanShift()
ms_115.fit(label_115_loss)

labels_115 = ms_115.labels_  # 第0类中的每个样本对应的簇号

n_clusters_115 = len(np.unique(labels_115))  # 第0类样本有几个簇

need_sampled_cluster_115 = []  # 需要下采样的簇号

cluster_to_num_115 = Counter(labels_115)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_115.keys():
    if cluster_to_num_115[k] > len(labels_115) // n_clusters_115:
        need_sampled_cluster_115.append(k)

need_sampled_losses_115 = [[] for _ in range(len(need_sampled_cluster_115))]

for i in range(len(need_sampled_cluster_115)):
    for j in range(len(labels_115)):
        if labels_115[j] == need_sampled_cluster_115[i]:
            need_sampled_losses_115[i].append(label_115_loss[j][0])

for j in range(len(labels_115)):
    if labels_115[j] not in need_sampled_cluster_115:
        selected_sampled_losses_115.append(label_115_loss[j][0])

for loss in need_sampled_losses_115:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_115.append(lo)
print(len(label_115_loss))
print(len(set(selected_sampled_losses_115)))

for loss in list(set(selected_sampled_losses_115)):
    for img in label_115_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第116类的采样=================================================
selected_sampled_losses_116 = []
ms_116 = MeanShift()
ms_116.fit(label_116_loss)

labels_116 = ms_116.labels_  # 第0类中的每个样本对应的簇号

n_clusters_116 = len(np.unique(labels_116))  # 第0类样本有几个簇

need_sampled_cluster_116 = []  # 需要下采样的簇号

cluster_to_num_116 = Counter(labels_116)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_116.keys():
    if cluster_to_num_116[k] > len(labels_116) // n_clusters_116:
        need_sampled_cluster_116.append(k)

need_sampled_losses_116 = [[] for _ in range(len(need_sampled_cluster_116))]

for i in range(len(need_sampled_cluster_116)):
    for j in range(len(labels_116)):
        if labels_116[j] == need_sampled_cluster_116[i]:
            need_sampled_losses_116[i].append(label_116_loss[j][0])

for j in range(len(labels_116)):
    if labels_116[j] not in need_sampled_cluster_116:
        selected_sampled_losses_116.append(label_116_loss[j][0])

for loss in need_sampled_losses_116:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_116.append(lo)
print(len(label_116_loss))
print(len(set(selected_sampled_losses_116)))

for loss in list(set(selected_sampled_losses_116)):
    for img in label_116_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第117类的采样=================================================
selected_sampled_losses_117 = []
ms_117 = MeanShift()
ms_117.fit(label_117_loss)

labels_117 = ms_117.labels_  # 第0类中的每个样本对应的簇号

n_clusters_117 = len(np.unique(labels_117))  # 第0类样本有几个簇

need_sampled_cluster_117 = []  # 需要下采样的簇号

cluster_to_num_117 = Counter(labels_117)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_117.keys():
    if cluster_to_num_117[k] > len(labels_117) // n_clusters_117:
        need_sampled_cluster_117.append(k)

need_sampled_losses_117 = [[] for _ in range(len(need_sampled_cluster_117))]

for i in range(len(need_sampled_cluster_117)):
    for j in range(len(labels_117)):
        if labels_117[j] == need_sampled_cluster_117[i]:
            need_sampled_losses_117[i].append(label_117_loss[j][0])

for j in range(len(labels_117)):
    if labels_117[j] not in need_sampled_cluster_117:
        selected_sampled_losses_117.append(label_117_loss[j][0])

for loss in need_sampled_losses_117:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_117.append(lo)
print(len(label_117_loss))
print(len(set(selected_sampled_losses_117)))

for loss in list(set(selected_sampled_losses_117)):
    for img in label_117_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第118类的采样=================================================
selected_sampled_losses_118 = []
ms_118 = MeanShift()
ms_118.fit(label_118_loss)

labels_118 = ms_118.labels_  # 第0类中的每个样本对应的簇号

n_clusters_118 = len(np.unique(labels_118))  # 第0类样本有几个簇

need_sampled_cluster_118 = []  # 需要下采样的簇号

cluster_to_num_118 = Counter(labels_118)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_118.keys():
    if cluster_to_num_118[k] > len(labels_118) // n_clusters_118:
        need_sampled_cluster_118.append(k)

need_sampled_losses_118 = [[] for _ in range(len(need_sampled_cluster_118))]

for i in range(len(need_sampled_cluster_118)):
    for j in range(len(labels_118)):
        if labels_118[j] == need_sampled_cluster_118[i]:
            need_sampled_losses_118[i].append(label_118_loss[j][0])

for j in range(len(labels_118)):
    if labels_118[j] not in need_sampled_cluster_118:
        selected_sampled_losses_118.append(label_118_loss[j][0])

for loss in need_sampled_losses_118:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_118.append(lo)
print(len(label_118_loss))
print(len(set(selected_sampled_losses_118)))

for loss in list(set(selected_sampled_losses_118)):
    for img in label_118_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第119类的采样=================================================
selected_sampled_losses_119 = []
ms_119 = MeanShift()
ms_119.fit(label_119_loss)

labels_119 = ms_119.labels_  # 第0类中的每个样本对应的簇号

n_clusters_119 = len(np.unique(labels_119))  # 第0类样本有几个簇

need_sampled_cluster_119 = []  # 需要下采样的簇号

cluster_to_num_119 = Counter(labels_119)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_119.keys():
    if cluster_to_num_119[k] > len(labels_119) // n_clusters_119:
        need_sampled_cluster_119.append(k)

need_sampled_losses_119 = [[] for _ in range(len(need_sampled_cluster_119))]

for i in range(len(need_sampled_cluster_119)):
    for j in range(len(labels_119)):
        if labels_119[j] == need_sampled_cluster_119[i]:
            need_sampled_losses_119[i].append(label_119_loss[j][0])

for j in range(len(labels_119)):
    if labels_119[j] not in need_sampled_cluster_119:
        selected_sampled_losses_119.append(label_119_loss[j][0])

for loss in need_sampled_losses_119:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_119.append(lo)
print(len(label_119_loss))
print(len(set(selected_sampled_losses_119)))
for loss in list(set(selected_sampled_losses_119)):
    for img in label_119_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))






# ==================================================针对第120类的采样=================================================

selected_sampled_losses_120 = []

ms_120 = MeanShift()
ms_120.fit(label_120_loss)

labels_120 = ms_120.labels_  # 第0类中的每个样本对应的簇号

n_clusters_120 = len(np.unique(labels_120))  # 第0类样本有几个簇

need_sampled_cluster_120 = []  # 需要下采样的簇号

cluster_to_num_120 = Counter(labels_120)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_120.keys():
    if cluster_to_num_120[k] > len(labels_120) // n_clusters_120:
        need_sampled_cluster_120.append(k)

need_sampled_losses_120 = [[] for _ in range(len(need_sampled_cluster_120))]

for i in range(len(need_sampled_cluster_120)):
    for j in range(len(labels_120)):
        if labels_120[j] == need_sampled_cluster_120[i]:
            need_sampled_losses_120[i].append(label_120_loss[j][0])

for j in range(len(labels_120)):
    if labels_120[j] not in need_sampled_cluster_120:
        selected_sampled_losses_120.append(label_120_loss[j][0])

for loss in need_sampled_losses_120:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_120.append(lo)
print(len(label_120_loss))
print(len(set(selected_sampled_losses_120)))

for loss in list(set(selected_sampled_losses_120)):
    for img in label_120_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第121类的采样=================================================
selected_sampled_losses_121 = []

ms_121 = MeanShift()
ms_121.fit(label_121_loss)

labels_121 = ms_121.labels_  # 第0类中的每个样本对应的簇号

n_clusters_121 = len(np.unique(labels_121))  # 第0类样本有几个簇

need_sampled_cluster_121 = []  # 需要下采样的簇号

cluster_to_num_121 = Counter(labels_121)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_121.keys():
    if cluster_to_num_121[k] > len(labels_121) // n_clusters_121:
        need_sampled_cluster_121.append(k)

need_sampled_losses_121 = [[] for _ in range(len(need_sampled_cluster_121))]

for i in range(len(need_sampled_cluster_121)):
    for j in range(len(labels_121)):
        if labels_121[j] == need_sampled_cluster_121[i]:
            need_sampled_losses_121[i].append(label_121_loss[j][0])

for j in range(len(labels_121)):
    if labels_121[j] not in need_sampled_cluster_121:
        selected_sampled_losses_121.append(label_121_loss[j][0])

for loss in need_sampled_losses_121:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_121.append(lo)
print(len(label_121_loss))
print(len(set(selected_sampled_losses_121)))

for loss in list(set(selected_sampled_losses_121)):
    for img in label_121_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第122类的采样=================================================
selected_sampled_losses_122 = []

ms_122 = MeanShift()
ms_122.fit(label_122_loss)

labels_122 = ms_122.labels_  # 第0类中的每个样本对应的簇号

n_clusters_122 = len(np.unique(labels_122))  # 第0类样本有几个簇

need_sampled_cluster_122 = []  # 需要下采样的簇号

cluster_to_num_122 = Counter(labels_122)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_122.keys():
    if cluster_to_num_122[k] > len(labels_122) // n_clusters_122:
        need_sampled_cluster_122.append(k)

need_sampled_losses_122 = [[] for _ in range(len(need_sampled_cluster_122))]

for i in range(len(need_sampled_cluster_122)):
    for j in range(len(labels_122)):
        if labels_122[j] == need_sampled_cluster_122[i]:
            need_sampled_losses_122[i].append(label_122_loss[j][0])

for j in range(len(labels_122)):
    if labels_122[j] not in need_sampled_cluster_122:
        selected_sampled_losses_122.append(label_122_loss[j][0])

for loss in need_sampled_losses_122:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_122.append(lo)
print(len(label_122_loss))
print(len(set(selected_sampled_losses_122)))

for loss in list(set(selected_sampled_losses_122)):
    for img in label_122_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第123类的采样=================================================
selected_sampled_losses_123 = []

ms_123 = MeanShift()
ms_123.fit(label_123_loss)

labels_123 = ms_123.labels_  # 第0类中的每个样本对应的簇号

n_clusters_123 = len(np.unique(labels_123))  # 第0类样本有几个簇

need_sampled_cluster_123 = []  # 需要下采样的簇号

cluster_to_num_123 = Counter(labels_123)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_123.keys():
    if cluster_to_num_123[k] > len(labels_123) // n_clusters_123:
        need_sampled_cluster_123.append(k)

need_sampled_losses_123 = [[] for _ in range(len(need_sampled_cluster_123))]

for i in range(len(need_sampled_cluster_123)):
    for j in range(len(labels_123)):
        if labels_123[j] == need_sampled_cluster_123[i]:
            need_sampled_losses_123[i].append(label_123_loss[j][0])

for j in range(len(labels_123)):
    if labels_123[j] not in need_sampled_cluster_123:
        selected_sampled_losses_123.append(label_123_loss[j][0])

for loss in need_sampled_losses_123:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_123.append(lo)
print(len(label_123_loss))
print(len(set(selected_sampled_losses_123)))

for loss in list(set(selected_sampled_losses_123)):
    for img in label_123_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第124类的采样=================================================
selected_sampled_losses_124 = []
ms_124 = MeanShift()
ms_124.fit(label_124_loss)

labels_124 = ms_124.labels_  # 第0类中的每个样本对应的簇号

n_clusters_124 = len(np.unique(labels_124))  # 第0类样本有几个簇

need_sampled_cluster_124 = []  # 需要下采样的簇号

cluster_to_num_124 = Counter(labels_124)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_124.keys():
    if cluster_to_num_124[k] > len(labels_124) // n_clusters_124:
        need_sampled_cluster_124.append(k)

need_sampled_losses_124 = [[] for _ in range(len(need_sampled_cluster_124))]

for i in range(len(need_sampled_cluster_124)):
    for j in range(len(labels_124)):
        if labels_124[j] == need_sampled_cluster_124[i]:
            need_sampled_losses_124[i].append(label_124_loss[j][0])

for j in range(len(labels_124)):
    if labels_124[j] not in need_sampled_cluster_124:
        selected_sampled_losses_124.append(label_124_loss[j][0])

for loss in need_sampled_losses_124:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_124.append(lo)
print(len(label_124_loss))
print(len(set(selected_sampled_losses_124)))

for loss in list(set(selected_sampled_losses_124)):
    for img in label_124_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第125类的采样=================================================
selected_sampled_losses_125 = []
ms_125 = MeanShift()
ms_125.fit(label_125_loss)

labels_125 = ms_125.labels_  # 第0类中的每个样本对应的簇号

n_clusters_125 = len(np.unique(labels_125))  # 第0类样本有几个簇

need_sampled_cluster_125 = []  # 需要下采样的簇号

cluster_to_num_125 = Counter(labels_125)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_125.keys():
    if cluster_to_num_125[k] > len(labels_125) // n_clusters_125:
        need_sampled_cluster_125.append(k)

need_sampled_losses_125 = [[] for _ in range(len(need_sampled_cluster_125))]

for i in range(len(need_sampled_cluster_125)):
    for j in range(len(labels_125)):
        if labels_125[j] == need_sampled_cluster_125[i]:
            need_sampled_losses_125[i].append(label_125_loss[j][0])

for j in range(len(labels_125)):
    if labels_125[j] not in need_sampled_cluster_125:
        selected_sampled_losses_125.append(label_125_loss[j][0])

for loss in need_sampled_losses_125:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_125.append(lo)
print(len(label_125_loss))
print(len(set(selected_sampled_losses_125)))

for loss in list(set(selected_sampled_losses_125)):
    for img in label_125_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第126类的采样=================================================
selected_sampled_losses_126 = []
ms_126 = MeanShift()
ms_126.fit(label_126_loss)

labels_126 = ms_126.labels_  # 第0类中的每个样本对应的簇号

n_clusters_126 = len(np.unique(labels_126))  # 第0类样本有几个簇

need_sampled_cluster_126 = []  # 需要下采样的簇号

cluster_to_num_126 = Counter(labels_126)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_126.keys():
    if cluster_to_num_126[k] > len(labels_126) // n_clusters_126:
        need_sampled_cluster_126.append(k)

need_sampled_losses_126 = [[] for _ in range(len(need_sampled_cluster_126))]

for i in range(len(need_sampled_cluster_126)):
    for j in range(len(labels_126)):
        if labels_126[j] == need_sampled_cluster_126[i]:
            need_sampled_losses_126[i].append(label_126_loss[j][0])

for j in range(len(labels_126)):
    if labels_126[j] not in need_sampled_cluster_126:
        selected_sampled_losses_126.append(label_126_loss[j][0])

for loss in need_sampled_losses_126:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_126.append(lo)
print(len(label_126_loss))
print(len(set(selected_sampled_losses_126)))

for loss in list(set(selected_sampled_losses_126)):
    for img in label_126_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第127类的采样=================================================
selected_sampled_losses_127 = []
ms_127 = MeanShift()
ms_127.fit(label_127_loss)

labels_127 = ms_127.labels_  # 第0类中的每个样本对应的簇号

n_clusters_127 = len(np.unique(labels_127))  # 第0类样本有几个簇

need_sampled_cluster_127 = []  # 需要下采样的簇号

cluster_to_num_127 = Counter(labels_127)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_127.keys():
    if cluster_to_num_127[k] > len(labels_127) // n_clusters_127:
        need_sampled_cluster_127.append(k)

need_sampled_losses_127 = [[] for _ in range(len(need_sampled_cluster_127))]

for i in range(len(need_sampled_cluster_127)):
    for j in range(len(labels_127)):
        if labels_127[j] == need_sampled_cluster_127[i]:
            need_sampled_losses_127[i].append(label_127_loss[j][0])

for j in range(len(labels_127)):
    if labels_127[j] not in need_sampled_cluster_127:
        selected_sampled_losses_127.append(label_127_loss[j][0])

for loss in need_sampled_losses_127:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_127.append(lo)
print(len(label_127_loss))
print(len(set(selected_sampled_losses_127)))

for loss in list(set(selected_sampled_losses_127)):
    for img in label_127_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第128类的采样=================================================
selected_sampled_losses_128 = []
ms_128 = MeanShift()
ms_128.fit(label_128_loss)

labels_128 = ms_128.labels_  # 第0类中的每个样本对应的簇号

n_clusters_128 = len(np.unique(labels_128))  # 第0类样本有几个簇

need_sampled_cluster_128 = []  # 需要下采样的簇号

cluster_to_num_128 = Counter(labels_128)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_128.keys():
    if cluster_to_num_128[k] > len(labels_128) // n_clusters_128:
        need_sampled_cluster_128.append(k)

need_sampled_losses_128 = [[] for _ in range(len(need_sampled_cluster_128))]

for i in range(len(need_sampled_cluster_128)):
    for j in range(len(labels_128)):
        if labels_128[j] == need_sampled_cluster_128[i]:
            need_sampled_losses_128[i].append(label_128_loss[j][0])

for j in range(len(labels_128)):
    if labels_128[j] not in need_sampled_cluster_128:
        selected_sampled_losses_128.append(label_128_loss[j][0])

for loss in need_sampled_losses_128:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_128.append(lo)
print(len(label_128_loss))
print(len(set(selected_sampled_losses_128)))

for loss in list(set(selected_sampled_losses_128)):
    for img in label_128_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第129类的采样=================================================
selected_sampled_losses_129 = []
ms_129 = MeanShift()
ms_129.fit(label_129_loss)

labels_129 = ms_129.labels_  # 第0类中的每个样本对应的簇号

n_clusters_129 = len(np.unique(labels_129))  # 第0类样本有几个簇

need_sampled_cluster_129 = []  # 需要下采样的簇号

cluster_to_num_129 = Counter(labels_129)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_129.keys():
    if cluster_to_num_129[k] > len(labels_129) // n_clusters_129:
        need_sampled_cluster_129.append(k)

need_sampled_losses_129 = [[] for _ in range(len(need_sampled_cluster_129))]

for i in range(len(need_sampled_cluster_129)):
    for j in range(len(labels_129)):
        if labels_129[j] == need_sampled_cluster_129[i]:
            need_sampled_losses_129[i].append(label_129_loss[j][0])

for j in range(len(labels_129)):
    if labels_129[j] not in need_sampled_cluster_129:
        selected_sampled_losses_129.append(label_129_loss[j][0])

for loss in need_sampled_losses_129:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_129.append(lo)
print(len(label_129_loss))
print(len(set(selected_sampled_losses_129)))

for loss in list(set(selected_sampled_losses_129)):
    for img in label_129_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第130类的采样=================================================
selected_sampled_losses_130 = []
ms_130 = MeanShift()
ms_130.fit(label_130_loss)

labels_130 = ms_130.labels_  # 第0类中的每个样本对应的簇号

n_clusters_130 = len(np.unique(labels_130))  # 第0类样本有几个簇

need_sampled_cluster_130 = []  # 需要下采样的簇号

cluster_to_num_130 = Counter(labels_130)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_130.keys():
    if cluster_to_num_130[k] > len(labels_130) // n_clusters_130:
        need_sampled_cluster_130.append(k)

need_sampled_losses_130 = [[] for _ in range(len(need_sampled_cluster_130))]

for i in range(len(need_sampled_cluster_130)):
    for j in range(len(labels_130)):
        if labels_130[j] == need_sampled_cluster_130[i]:
            need_sampled_losses_130[i].append(label_130_loss[j][0])

for j in range(len(labels_130)):
    if labels_130[j] not in need_sampled_cluster_130:
        selected_sampled_losses_130.append(label_130_loss[j][0])

for loss in need_sampled_losses_130:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_130.append(lo)
print(len(label_130_loss))
print(len(set(selected_sampled_losses_130)))

for loss in list(set(selected_sampled_losses_130)):
    for img in label_130_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第131类的采样=================================================
selected_sampled_losses_131 = []

ms_131 = MeanShift()
ms_131.fit(label_131_loss)

labels_131 = ms_131.labels_  # 第0类中的每个样本对应的簇号

n_clusters_131 = len(np.unique(labels_131))  # 第0类样本有几个簇

need_sampled_cluster_131 = []  # 需要下采样的簇号

cluster_to_num_131 = Counter(labels_131)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_131.keys():
    if cluster_to_num_131[k] > len(labels_131) // n_clusters_131:
        need_sampled_cluster_131.append(k)

need_sampled_losses_131 = [[] for _ in range(len(need_sampled_cluster_131))]

for i in range(len(need_sampled_cluster_131)):
    for j in range(len(labels_131)):
        if labels_131[j] == need_sampled_cluster_131[i]:
            need_sampled_losses_131[i].append(label_131_loss[j][0])

for j in range(len(labels_131)):
    if labels_131[j] not in need_sampled_cluster_131:
        selected_sampled_losses_131.append(label_131_loss[j][0])

for loss in need_sampled_losses_131:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_131.append(lo)
print(len(label_131_loss))
print(len(set(selected_sampled_losses_131)))

for loss in list(set(selected_sampled_losses_131)):
    for img in label_131_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第132类的采样=================================================
selected_sampled_losses_132 = []
ms_132 = MeanShift()
ms_132.fit(label_132_loss)

labels_132 = ms_132.labels_  # 第0类中的每个样本对应的簇号

n_clusters_132 = len(np.unique(labels_132))  # 第0类样本有几个簇

need_sampled_cluster_132 = []  # 需要下采样的簇号

cluster_to_num_132 = Counter(labels_132)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_132.keys():
    if cluster_to_num_132[k] > len(labels_132) // n_clusters_132:
        need_sampled_cluster_132.append(k)

need_sampled_losses_132 = [[] for _ in range(len(need_sampled_cluster_132))]

for i in range(len(need_sampled_cluster_132)):
    for j in range(len(labels_132)):
        if labels_132[j] == need_sampled_cluster_132[i]:
            need_sampled_losses_132[i].append(label_132_loss[j][0])

for j in range(len(labels_132)):
    if labels_132[j] not in need_sampled_cluster_132:
        selected_sampled_losses_132.append(label_132_loss[j][0])

for loss in need_sampled_losses_132:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_132.append(lo)
print(len(label_132_loss))
print(len(set(selected_sampled_losses_132)))

for loss in list(set(selected_sampled_losses_132)):
    for img in label_132_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第133类的采样=================================================
selected_sampled_losses_133 = []
ms_133 = MeanShift()
ms_133.fit(label_133_loss)

labels_133 = ms_133.labels_  # 第0类中的每个样本对应的簇号

n_clusters_133 = len(np.unique(labels_133))  # 第0类样本有几个簇

need_sampled_cluster_133 = []  # 需要下采样的簇号

cluster_to_num_133 = Counter(labels_133)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_133.keys():
    if cluster_to_num_133[k] > len(labels_133) // n_clusters_133:
        need_sampled_cluster_133.append(k)

need_sampled_losses_133 = [[] for _ in range(len(need_sampled_cluster_133))]

for i in range(len(need_sampled_cluster_133)):
    for j in range(len(labels_133)):
        if labels_133[j] == need_sampled_cluster_133[i]:
            need_sampled_losses_133[i].append(label_133_loss[j][0])

for j in range(len(labels_133)):
    if labels_133[j] not in need_sampled_cluster_133:
        selected_sampled_losses_133.append(label_133_loss[j][0])

for loss in need_sampled_losses_133:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_133.append(lo)
print(len(label_133_loss))
print(len(set(selected_sampled_losses_133)))

for loss in list(set(selected_sampled_losses_133)):
    for img in label_133_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第134类的采样=================================================
selected_sampled_losses_134 = []
ms_134 = MeanShift()
ms_134.fit(label_134_loss)

labels_134 = ms_134.labels_  # 第0类中的每个样本对应的簇号

n_clusters_134 = len(np.unique(labels_134))  # 第0类样本有几个簇

need_sampled_cluster_134 = []  # 需要下采样的簇号

cluster_to_num_134 = Counter(labels_134)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_134.keys():
    if cluster_to_num_134[k] > len(labels_134) // n_clusters_134:
        need_sampled_cluster_134.append(k)

need_sampled_losses_134 = [[] for _ in range(len(need_sampled_cluster_134))]

for i in range(len(need_sampled_cluster_134)):
    for j in range(len(labels_134)):
        if labels_134[j] == need_sampled_cluster_134[i]:
            need_sampled_losses_134[i].append(label_134_loss[j][0])

for j in range(len(labels_134)):
    if labels_134[j] not in need_sampled_cluster_134:
        selected_sampled_losses_134.append(label_134_loss[j][0])

for loss in need_sampled_losses_134:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_134.append(lo)
print(len(label_134_loss))
print(len(set(selected_sampled_losses_134)))

for loss in list(set(selected_sampled_losses_134)):
    for img in label_134_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第135类的采样=================================================
selected_sampled_losses_135 = []
ms_135 = MeanShift()
ms_135.fit(label_135_loss)

labels_135 = ms_135.labels_  # 第0类中的每个样本对应的簇号

n_clusters_135 = len(np.unique(labels_135))  # 第0类样本有几个簇

need_sampled_cluster_135 = []  # 需要下采样的簇号

cluster_to_num_135 = Counter(labels_135)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_135.keys():
    if cluster_to_num_135[k] > len(labels_135) // n_clusters_135:
        need_sampled_cluster_135.append(k)

need_sampled_losses_135 = [[] for _ in range(len(need_sampled_cluster_135))]

for i in range(len(need_sampled_cluster_135)):
    for j in range(len(labels_135)):
        if labels_135[j] == need_sampled_cluster_135[i]:
            need_sampled_losses_135[i].append(label_135_loss[j][0])

for j in range(len(labels_135)):
    if labels_135[j] not in need_sampled_cluster_135:
        selected_sampled_losses_135.append(label_135_loss[j][0])

for loss in need_sampled_losses_135:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_135.append(lo)
print(len(label_135_loss))
print(len(set(selected_sampled_losses_135)))

for loss in list(set(selected_sampled_losses_135)):
    for img in label_135_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第136类的采样=================================================
selected_sampled_losses_136 = []
ms_136 = MeanShift()
ms_136.fit(label_136_loss)

labels_136 = ms_136.labels_  # 第0类中的每个样本对应的簇号

n_clusters_136 = len(np.unique(labels_136))  # 第0类样本有几个簇

need_sampled_cluster_136 = []  # 需要下采样的簇号

cluster_to_num_136 = Counter(labels_136)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_136.keys():
    if cluster_to_num_136[k] > len(labels_136) // n_clusters_136:
        need_sampled_cluster_136.append(k)

need_sampled_losses_136 = [[] for _ in range(len(need_sampled_cluster_136))]

for i in range(len(need_sampled_cluster_136)):
    for j in range(len(labels_136)):
        if labels_136[j] == need_sampled_cluster_136[i]:
            need_sampled_losses_136[i].append(label_136_loss[j][0])

for j in range(len(labels_136)):
    if labels_136[j] not in need_sampled_cluster_136:
        selected_sampled_losses_136.append(label_136_loss[j][0])

for loss in need_sampled_losses_136:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_136.append(lo)
print(len(label_136_loss))
print(len(set(selected_sampled_losses_136)))

for loss in list(set(selected_sampled_losses_136)):
    for img in label_136_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第137类的采样=================================================
selected_sampled_losses_137 = []
ms_137 = MeanShift()
ms_137.fit(label_137_loss)

labels_137 = ms_137.labels_  # 第0类中的每个样本对应的簇号

n_clusters_137 = len(np.unique(labels_137))  # 第0类样本有几个簇

need_sampled_cluster_137 = []  # 需要下采样的簇号

cluster_to_num_137 = Counter(labels_137)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_137.keys():
    if cluster_to_num_137[k] > len(labels_137) // n_clusters_137:
        need_sampled_cluster_137.append(k)

need_sampled_losses_137 = [[] for _ in range(len(need_sampled_cluster_137))]

for i in range(len(need_sampled_cluster_137)):
    for j in range(len(labels_137)):
        if labels_137[j] == need_sampled_cluster_137[i]:
            need_sampled_losses_137[i].append(label_137_loss[j][0])

for j in range(len(labels_137)):
    if labels_137[j] not in need_sampled_cluster_137:
        selected_sampled_losses_137.append(label_137_loss[j][0])

for loss in need_sampled_losses_137:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_137.append(lo)
print(len(label_137_loss))
print(len(set(selected_sampled_losses_137)))

for loss in list(set(selected_sampled_losses_137)):
    for img in label_137_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第138类的采样=================================================
selected_sampled_losses_138 = []
ms_138 = MeanShift()
ms_138.fit(label_138_loss)

labels_138 = ms_138.labels_  # 第0类中的每个样本对应的簇号

n_clusters_138 = len(np.unique(labels_138))  # 第0类样本有几个簇

need_sampled_cluster_138 = []  # 需要下采样的簇号

cluster_to_num_138 = Counter(labels_138)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_138.keys():
    if cluster_to_num_138[k] > len(labels_138) // n_clusters_138:
        need_sampled_cluster_138.append(k)

need_sampled_losses_138 = [[] for _ in range(len(need_sampled_cluster_138))]

for i in range(len(need_sampled_cluster_138)):
    for j in range(len(labels_138)):
        if labels_138[j] == need_sampled_cluster_138[i]:
            need_sampled_losses_138[i].append(label_138_loss[j][0])

for j in range(len(labels_138)):
    if labels_138[j] not in need_sampled_cluster_138:
        selected_sampled_losses_138.append(label_138_loss[j][0])

for loss in need_sampled_losses_138:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_138.append(lo)
print(len(label_138_loss))
print(len(set(selected_sampled_losses_138)))

for loss in list(set(selected_sampled_losses_138)):
    for img in label_138_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第139类的采样=================================================
selected_sampled_losses_139 = []
ms_139 = MeanShift()
ms_139.fit(label_139_loss)

labels_139 = ms_139.labels_  # 第0类中的每个样本对应的簇号

n_clusters_139 = len(np.unique(labels_139))  # 第0类样本有几个簇

need_sampled_cluster_139 = []  # 需要下采样的簇号

cluster_to_num_139 = Counter(labels_139)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_139.keys():
    if cluster_to_num_139[k] > len(labels_139) // n_clusters_139:
        need_sampled_cluster_139.append(k)

need_sampled_losses_139 = [[] for _ in range(len(need_sampled_cluster_139))]

for i in range(len(need_sampled_cluster_139)):
    for j in range(len(labels_139)):
        if labels_139[j] == need_sampled_cluster_139[i]:
            need_sampled_losses_139[i].append(label_139_loss[j][0])

for j in range(len(labels_139)):
    if labels_139[j] not in need_sampled_cluster_139:
        selected_sampled_losses_139.append(label_139_loss[j][0])

for loss in need_sampled_losses_139:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_139.append(lo)
print(len(label_139_loss))
print(len(set(selected_sampled_losses_139)))
for loss in list(set(selected_sampled_losses_139)):
    for img in label_139_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))

# ==================================================针对第140类的采样=================================================

selected_sampled_losses_140 = []

ms_140 = MeanShift()
ms_140.fit(label_140_loss)

labels_140 = ms_140.labels_  # 第_140类中的每个样本对应的簇号

n_clusters_140 = len(np.unique(labels_140))  # 第_140类样本有几个簇

need_sampled_cluster_140 = []  # 需要下采样的簇号

cluster_to_num_140 = Counter(labels_140)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_140.keys():
    if cluster_to_num_140[k] > len(labels_140) // n_clusters_140:
        need_sampled_cluster_140.append(k)

need_sampled_losses_140 = [[] for _ in range(len(need_sampled_cluster_140))]

for i in range(len(need_sampled_cluster_140)):
    for j in range(len(labels_140)):
        if labels_140[j] == need_sampled_cluster_140[i]:
            need_sampled_losses_140[i].append(label_140_loss[j][0])

for j in range(len(labels_140)):
    if labels_140[j] not in need_sampled_cluster_140:
        selected_sampled_losses_140.append(label_140_loss[j][0])

for loss in need_sampled_losses_140:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_140.append(lo)
print(len(label_140_loss))
print(len(set(selected_sampled_losses_140)))

for loss in list(set(selected_sampled_losses_140)):
    for img in label_140_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第141类的采样=================================================
selected_sampled_losses_141 = []

ms_141 = MeanShift()
ms_141.fit(label_141_loss)

labels_141 = ms_141.labels_  # 第0类中的每个样本对应的簇号

n_clusters_141 = len(np.unique(labels_141))  # 第0类样本有几个簇

need_sampled_cluster_141 = []  # 需要下采样的簇号

cluster_to_num_141 = Counter(labels_141)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_141.keys():
    if cluster_to_num_141[k] > len(labels_141) // n_clusters_141:
        need_sampled_cluster_141.append(k)

need_sampled_losses_141 = [[] for _ in range(len(need_sampled_cluster_141))]

for i in range(len(need_sampled_cluster_141)):
    for j in range(len(labels_141)):
        if labels_141[j] == need_sampled_cluster_141[i]:
            need_sampled_losses_141[i].append(label_141_loss[j][0])

for j in range(len(labels_141)):
    if labels_141[j] not in need_sampled_cluster_141:
        selected_sampled_losses_141.append(label_141_loss[j][0])

for loss in need_sampled_losses_141:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_141.append(lo)
print(len(label_141_loss))
print(len(set(selected_sampled_losses_141)))

for loss in list(set(selected_sampled_losses_141)):
    for img in label_141_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第142类的采样=================================================
selected_sampled_losses_142 = []

ms_142 = MeanShift()
ms_142.fit(label_142_loss)

labels_142 = ms_142.labels_  # 第0类中的每个样本对应的簇号

n_clusters_142 = len(np.unique(labels_142))  # 第0类样本有几个簇

need_sampled_cluster_142 = []  # 需要下采样的簇号

cluster_to_num_142 = Counter(labels_142)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_142.keys():
    if cluster_to_num_142[k] > len(labels_142) // n_clusters_142:
        need_sampled_cluster_142.append(k)

need_sampled_losses_142 = [[] for _ in range(len(need_sampled_cluster_142))]

for i in range(len(need_sampled_cluster_142)):
    for j in range(len(labels_142)):
        if labels_142[j] == need_sampled_cluster_142[i]:
            need_sampled_losses_142[i].append(label_142_loss[j][0])

for j in range(len(labels_142)):
    if labels_142[j] not in need_sampled_cluster_142:
        selected_sampled_losses_142.append(label_142_loss[j][0])

for loss in need_sampled_losses_142:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_142.append(lo)
print(len(label_142_loss))
print(len(set(selected_sampled_losses_142)))

for loss in list(set(selected_sampled_losses_142)):
    for img in label_142_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第143类的采样=================================================
selected_sampled_losses_143 = []

ms_143 = MeanShift()
ms_143.fit(label_143_loss)

labels_143 = ms_143.labels_  # 第0类中的每个样本对应的簇号

n_clusters_143 = len(np.unique(labels_143))  # 第0类样本有几个簇

need_sampled_cluster_143 = []  # 需要下采样的簇号

cluster_to_num_143 = Counter(labels_143)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_143.keys():
    if cluster_to_num_143[k] > len(labels_143) // n_clusters_143:
        need_sampled_cluster_143.append(k)

need_sampled_losses_143 = [[] for _ in range(len(need_sampled_cluster_143))]

for i in range(len(need_sampled_cluster_143)):
    for j in range(len(labels_143)):
        if labels_143[j] == need_sampled_cluster_143[i]:
            need_sampled_losses_143[i].append(label_143_loss[j][0])

for j in range(len(labels_143)):
    if labels_143[j] not in need_sampled_cluster_143:
        selected_sampled_losses_143.append(label_143_loss[j][0])

for loss in need_sampled_losses_143:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_143.append(lo)
print(len(label_143_loss))
print(len(set(selected_sampled_losses_143)))

for loss in list(set(selected_sampled_losses_143)):
    for img in label_143_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第144类的采样=================================================
selected_sampled_losses_144 = []
ms_144 = MeanShift()
ms_144.fit(label_144_loss)

labels_144 = ms_144.labels_  # 第0类中的每个样本对应的簇号

n_clusters_144 = len(np.unique(labels_144))  # 第0类样本有几个簇

need_sampled_cluster_144 = []  # 需要下采样的簇号

cluster_to_num_144 = Counter(labels_144)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_144.keys():
    if cluster_to_num_144[k] > len(labels_144) // n_clusters_144:
        need_sampled_cluster_144.append(k)

need_sampled_losses_144 = [[] for _ in range(len(need_sampled_cluster_144))]

for i in range(len(need_sampled_cluster_144)):
    for j in range(len(labels_144)):
        if labels_144[j] == need_sampled_cluster_144[i]:
            need_sampled_losses_144[i].append(label_144_loss[j][0])

for j in range(len(labels_144)):
    if labels_144[j] not in need_sampled_cluster_144:
        selected_sampled_losses_144.append(label_144_loss[j][0])

for loss in need_sampled_losses_144:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_144.append(lo)
print(len(label_144_loss))
print(len(set(selected_sampled_losses_144)))

for loss in list(set(selected_sampled_losses_144)):
    for img in label_144_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第145类的采样=================================================
selected_sampled_losses_145 = []
ms_145 = MeanShift()
ms_145.fit(label_145_loss)

labels_145 = ms_145.labels_  # 第0类中的每个样本对应的簇号

n_clusters_145 = len(np.unique(labels_145))  # 第0类样本有几个簇

need_sampled_cluster_145 = []  # 需要下采样的簇号

cluster_to_num_145 = Counter(labels_145)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_145.keys():
    if cluster_to_num_145[k] > len(labels_145) // n_clusters_145:
        need_sampled_cluster_145.append(k)

need_sampled_losses_145 = [[] for _ in range(len(need_sampled_cluster_145))]

for i in range(len(need_sampled_cluster_145)):
    for j in range(len(labels_145)):
        if labels_145[j] == need_sampled_cluster_145[i]:
            need_sampled_losses_145[i].append(label_145_loss[j][0])

for j in range(len(labels_145)):
    if labels_145[j] not in need_sampled_cluster_145:
        selected_sampled_losses_145.append(label_145_loss[j][0])

for loss in need_sampled_losses_145:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_145.append(lo)
print(len(label_145_loss))
print(len(set(selected_sampled_losses_145)))

for loss in list(set(selected_sampled_losses_145)):
    for img in label_145_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第146类的采样=================================================
selected_sampled_losses_146 = []
ms_146 = MeanShift()
ms_146.fit(label_146_loss)

labels_146 = ms_146.labels_  # 第0类中的每个样本对应的簇号

n_clusters_146 = len(np.unique(labels_146))  # 第0类样本有几个簇

need_sampled_cluster_146 = []  # 需要下采样的簇号

cluster_to_num_146 = Counter(labels_146)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_146.keys():
    if cluster_to_num_146[k] > len(labels_146) // n_clusters_146:
        need_sampled_cluster_146.append(k)

need_sampled_losses_146 = [[] for _ in range(len(need_sampled_cluster_146))]

for i in range(len(need_sampled_cluster_146)):
    for j in range(len(labels_146)):
        if labels_146[j] == need_sampled_cluster_146[i]:
            need_sampled_losses_146[i].append(label_146_loss[j][0])

for j in range(len(labels_146)):
    if labels_146[j] not in need_sampled_cluster_146:
        selected_sampled_losses_146.append(label_146_loss[j][0])

for loss in need_sampled_losses_146:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_146.append(lo)
print(len(label_146_loss))
print(len(set(selected_sampled_losses_146)))

for loss in list(set(selected_sampled_losses_146)):
    for img in label_146_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第147类的采样=================================================
selected_sampled_losses_147 = []
ms_147 = MeanShift()
ms_147.fit(label_147_loss)

labels_147 = ms_147.labels_  # 第0类中的每个样本对应的簇号

n_clusters_147 = len(np.unique(labels_147))  # 第0类样本有几个簇

need_sampled_cluster_147 = []  # 需要下采样的簇号

cluster_to_num_147 = Counter(labels_147)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_147.keys():
    if cluster_to_num_147[k] > len(labels_147) // n_clusters_147:
        need_sampled_cluster_147.append(k)

need_sampled_losses_147 = [[] for _ in range(len(need_sampled_cluster_147))]

for i in range(len(need_sampled_cluster_147)):
    for j in range(len(labels_147)):
        if labels_147[j] == need_sampled_cluster_147[i]:
            need_sampled_losses_147[i].append(label_147_loss[j][0])

for j in range(len(labels_147)):
    if labels_147[j] not in need_sampled_cluster_147:
        selected_sampled_losses_147.append(label_147_loss[j][0])

for loss in need_sampled_losses_147:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_147.append(lo)
print(len(label_147_loss))
print(len(set(selected_sampled_losses_147)))

for loss in list(set(selected_sampled_losses_147)):
    for img in label_147_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第148类的采样=================================================
selected_sampled_losses_148 = []
ms_148 = MeanShift()
ms_148.fit(label_148_loss)

labels_148 = ms_148.labels_  # 第0类中的每个样本对应的簇号

n_clusters_148 = len(np.unique(labels_148))  # 第0类样本有几个簇

need_sampled_cluster_148 = []  # 需要下采样的簇号

cluster_to_num_148 = Counter(labels_148)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_148.keys():
    if cluster_to_num_148[k] > len(labels_148) // n_clusters_148:
        need_sampled_cluster_148.append(k)

need_sampled_losses_148 = [[] for _ in range(len(need_sampled_cluster_148))]

for i in range(len(need_sampled_cluster_148)):
    for j in range(len(labels_148)):
        if labels_148[j] == need_sampled_cluster_148[i]:
            need_sampled_losses_148[i].append(label_148_loss[j][0])

for j in range(len(labels_148)):
    if labels_148[j] not in need_sampled_cluster_148:
        selected_sampled_losses_148.append(label_148_loss[j][0])

for loss in need_sampled_losses_148:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_148.append(lo)
print(len(label_148_loss))
print(len(set(selected_sampled_losses_148)))

for loss in list(set(selected_sampled_losses_148)):
    for img in label_148_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第149类的采样=================================================
selected_sampled_losses_149 = []
ms_149 = MeanShift()
ms_149.fit(label_149_loss)

labels_149 = ms_149.labels_  # 第0类中的每个样本对应的簇号

n_clusters_149 = len(np.unique(labels_149))  # 第0类样本有几个簇

need_sampled_cluster_149 = []  # 需要下采样的簇号

cluster_to_num_149 = Counter(labels_149)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_149.keys():
    if cluster_to_num_149[k] > len(labels_149) // n_clusters_149:
        need_sampled_cluster_149.append(k)

need_sampled_losses_149 = [[] for _ in range(len(need_sampled_cluster_149))]

for i in range(len(need_sampled_cluster_149)):
    for j in range(len(labels_149)):
        if labels_149[j] == need_sampled_cluster_149[i]:
            need_sampled_losses_149[i].append(label_149_loss[j][0])

for j in range(len(labels_149)):
    if labels_149[j] not in need_sampled_cluster_149:
        selected_sampled_losses_149.append(label_149_loss[j][0])

for loss in need_sampled_losses_149:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_149.append(lo)
print(len(label_149_loss))
print(len(set(selected_sampled_losses_149)))

for loss in list(set(selected_sampled_losses_149)):
    for img in label_149_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第150类的采样=================================================
selected_sampled_losses_150 = []
ms_150 = MeanShift()
ms_150.fit(label_150_loss)

labels_150 = ms_150.labels_  # 第0类中的每个样本对应的簇号

n_clusters_150 = len(np.unique(labels_150))  # 第0类样本有几个簇

need_sampled_cluster_150 = []  # 需要下采样的簇号

cluster_to_num_150 = Counter(labels_150)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_150.keys():
    if cluster_to_num_150[k] > len(labels_150) // n_clusters_150:
        need_sampled_cluster_150.append(k)

need_sampled_losses_150 = [[] for _ in range(len(need_sampled_cluster_150))]

for i in range(len(need_sampled_cluster_150)):
    for j in range(len(labels_150)):
        if labels_150[j] == need_sampled_cluster_150[i]:
            need_sampled_losses_150[i].append(label_150_loss[j][0])

for j in range(len(labels_150)):
    if labels_150[j] not in need_sampled_cluster_150:
        selected_sampled_losses_150.append(label_150_loss[j][0])

for loss in need_sampled_losses_150:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_150.append(lo)
print(len(label_150_loss))
print(len(set(selected_sampled_losses_150)))

for loss in list(set(selected_sampled_losses_150)):
    for img in label_150_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第151类的采样=================================================
selected_sampled_losses_151 = []

ms_151 = MeanShift()
ms_151.fit(label_151_loss)

labels_151 = ms_151.labels_  # 第0类中的每个样本对应的簇号

n_clusters_151 = len(np.unique(labels_151))  # 第0类样本有几个簇

need_sampled_cluster_151 = []  # 需要下采样的簇号

cluster_to_num_151 = Counter(labels_151)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_151.keys():
    if cluster_to_num_151[k] > len(labels_151) // n_clusters_151:
        need_sampled_cluster_151.append(k)

need_sampled_losses_151 = [[] for _ in range(len(need_sampled_cluster_151))]

for i in range(len(need_sampled_cluster_151)):
    for j in range(len(labels_151)):
        if labels_151[j] == need_sampled_cluster_151[i]:
            need_sampled_losses_151[i].append(label_151_loss[j][0])

for j in range(len(labels_151)):
    if labels_151[j] not in need_sampled_cluster_151:
        selected_sampled_losses_151.append(label_151_loss[j][0])

for loss in need_sampled_losses_151:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_151.append(lo)
print(len(label_151_loss))
print(len(set(selected_sampled_losses_151)))

for loss in list(set(selected_sampled_losses_151)):
    for img in label_151_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第152类的采样=================================================
selected_sampled_losses_152 = []
ms_152 = MeanShift()
ms_152.fit(label_152_loss)

labels_152 = ms_152.labels_  # 第0类中的每个样本对应的簇号

n_clusters_152 = len(np.unique(labels_152))  # 第0类样本有几个簇

need_sampled_cluster_152 = []  # 需要下采样的簇号

cluster_to_num_152 = Counter(labels_152)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_152.keys():
    if cluster_to_num_152[k] > len(labels_152) // n_clusters_152:
        need_sampled_cluster_152.append(k)

need_sampled_losses_152 = [[] for _ in range(len(need_sampled_cluster_152))]

for i in range(len(need_sampled_cluster_152)):
    for j in range(len(labels_152)):
        if labels_152[j] == need_sampled_cluster_152[i]:
            need_sampled_losses_152[i].append(label_152_loss[j][0])

for j in range(len(labels_152)):
    if labels_152[j] not in need_sampled_cluster_152:
        selected_sampled_losses_152.append(label_152_loss[j][0])

for loss in need_sampled_losses_152:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_152.append(lo)
print(len(label_152_loss))
print(len(set(selected_sampled_losses_152)))

for loss in list(set(selected_sampled_losses_152)):
    for img in label_152_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第153类的采样=================================================
selected_sampled_losses_153 = []
ms_153 = MeanShift()
ms_153.fit(label_153_loss)

labels_153 = ms_153.labels_  # 第0类中的每个样本对应的簇号

n_clusters_153 = len(np.unique(labels_153))  # 第0类样本有几个簇

need_sampled_cluster_153 = []  # 需要下采样的簇号

cluster_to_num_153 = Counter(labels_153)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_153.keys():
    if cluster_to_num_153[k] > len(labels_153) // n_clusters_153:
        need_sampled_cluster_153.append(k)

need_sampled_losses_153 = [[] for _ in range(len(need_sampled_cluster_153))]

for i in range(len(need_sampled_cluster_153)):
    for j in range(len(labels_153)):
        if labels_153[j] == need_sampled_cluster_153[i]:
            need_sampled_losses_153[i].append(label_153_loss[j][0])

for j in range(len(labels_153)):
    if labels_153[j] not in need_sampled_cluster_153:
        selected_sampled_losses_153.append(label_153_loss[j][0])

for loss in need_sampled_losses_153:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_153.append(lo)
print(len(label_153_loss))
print(len(set(selected_sampled_losses_153)))

for loss in list(set(selected_sampled_losses_153)):
    for img in label_153_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第154类的采样=================================================
selected_sampled_losses_154 = []
ms_154 = MeanShift()
ms_154.fit(label_154_loss)

labels_154 = ms_154.labels_  # 第0类中的每个样本对应的簇号

n_clusters_154 = len(np.unique(labels_154))  # 第0类样本有几个簇

need_sampled_cluster_154 = []  # 需要下采样的簇号

cluster_to_num_154 = Counter(labels_154)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_154.keys():
    if cluster_to_num_154[k] > len(labels_154) // n_clusters_154:
        need_sampled_cluster_154.append(k)

need_sampled_losses_154 = [[] for _ in range(len(need_sampled_cluster_154))]

for i in range(len(need_sampled_cluster_154)):
    for j in range(len(labels_154)):
        if labels_154[j] == need_sampled_cluster_154[i]:
            need_sampled_losses_154[i].append(label_154_loss[j][0])

for j in range(len(labels_154)):
    if labels_154[j] not in need_sampled_cluster_154:
        selected_sampled_losses_154.append(label_154_loss[j][0])

for loss in need_sampled_losses_154:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_154.append(lo)
print(len(label_154_loss))
print(len(set(selected_sampled_losses_154)))

for loss in list(set(selected_sampled_losses_154)):
    for img in label_154_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第155类的采样=================================================
selected_sampled_losses_155 = []
ms_155 = MeanShift()
ms_155.fit(label_155_loss)

labels_155 = ms_155.labels_  # 第0类中的每个样本对应的簇号

n_clusters_155 = len(np.unique(labels_155))  # 第0类样本有几个簇

need_sampled_cluster_155 = []  # 需要下采样的簇号

cluster_to_num_155 = Counter(labels_155)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_155.keys():
    if cluster_to_num_155[k] > len(labels_155) // n_clusters_155:
        need_sampled_cluster_155.append(k)

need_sampled_losses_155 = [[] for _ in range(len(need_sampled_cluster_155))]

for i in range(len(need_sampled_cluster_155)):
    for j in range(len(labels_155)):
        if labels_155[j] == need_sampled_cluster_155[i]:
            need_sampled_losses_155[i].append(label_155_loss[j][0])

for j in range(len(labels_155)):
    if labels_155[j] not in need_sampled_cluster_155:
        selected_sampled_losses_155.append(label_155_loss[j][0])

for loss in need_sampled_losses_155:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_155.append(lo)
print(len(label_155_loss))
print(len(set(selected_sampled_losses_155)))

for loss in list(set(selected_sampled_losses_155)):
    for img in label_155_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第156类的采样=================================================
selected_sampled_losses_156 = []
ms_156 = MeanShift()
ms_156.fit(label_156_loss)

labels_156 = ms_156.labels_  # 第0类中的每个样本对应的簇号

n_clusters_156 = len(np.unique(labels_156))  # 第0类样本有几个簇

need_sampled_cluster_156 = []  # 需要下采样的簇号

cluster_to_num_156 = Counter(labels_156)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_156.keys():
    if cluster_to_num_156[k] > len(labels_156) // n_clusters_156:
        need_sampled_cluster_156.append(k)

need_sampled_losses_156 = [[] for _ in range(len(need_sampled_cluster_156))]

for i in range(len(need_sampled_cluster_156)):
    for j in range(len(labels_156)):
        if labels_156[j] == need_sampled_cluster_156[i]:
            need_sampled_losses_156[i].append(label_156_loss[j][0])

for j in range(len(labels_156)):
    if labels_156[j] not in need_sampled_cluster_156:
        selected_sampled_losses_156.append(label_156_loss[j][0])

for loss in need_sampled_losses_156:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_156.append(lo)
print(len(label_156_loss))
print(len(set(selected_sampled_losses_156)))

for loss in list(set(selected_sampled_losses_156)):
    for img in label_156_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第157类的采样=================================================
selected_sampled_losses_157 = []
ms_157 = MeanShift()
ms_157.fit(label_157_loss)

labels_157 = ms_157.labels_  # 第0类中的每个样本对应的簇号

n_clusters_157 = len(np.unique(labels_157))  # 第0类样本有几个簇

need_sampled_cluster_157 = []  # 需要下采样的簇号

cluster_to_num_157 = Counter(labels_157)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_157.keys():
    if cluster_to_num_157[k] > len(labels_157) // n_clusters_157:
        need_sampled_cluster_157.append(k)

need_sampled_losses_157 = [[] for _ in range(len(need_sampled_cluster_157))]

for i in range(len(need_sampled_cluster_157)):
    for j in range(len(labels_157)):
        if labels_157[j] == need_sampled_cluster_157[i]:
            need_sampled_losses_157[i].append(label_157_loss[j][0])

for j in range(len(labels_157)):
    if labels_157[j] not in need_sampled_cluster_157:
        selected_sampled_losses_157.append(label_157_loss[j][0])

for loss in need_sampled_losses_157:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_157.append(lo)
print(len(label_157_loss))
print(len(set(selected_sampled_losses_157)))

for loss in list(set(selected_sampled_losses_157)):
    for img in label_157_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第158类的采样=================================================
selected_sampled_losses_158 = []
ms_158 = MeanShift()
ms_158.fit(label_158_loss)

labels_158 = ms_158.labels_  # 第0类中的每个样本对应的簇号

n_clusters_158 = len(np.unique(labels_158))  # 第0类样本有几个簇

need_sampled_cluster_158 = []  # 需要下采样的簇号

cluster_to_num_158 = Counter(labels_158)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_158.keys():
    if cluster_to_num_158[k] > len(labels_158) // n_clusters_158:
        need_sampled_cluster_158.append(k)

need_sampled_losses_158 = [[] for _ in range(len(need_sampled_cluster_158))]

for i in range(len(need_sampled_cluster_158)):
    for j in range(len(labels_158)):
        if labels_158[j] == need_sampled_cluster_158[i]:
            need_sampled_losses_158[i].append(label_158_loss[j][0])

for j in range(len(labels_158)):
    if labels_158[j] not in need_sampled_cluster_158:
        selected_sampled_losses_158.append(label_158_loss[j][0])

for loss in need_sampled_losses_158:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_158.append(lo)
print(len(label_158_loss))
print(len(set(selected_sampled_losses_158)))

for loss in list(set(selected_sampled_losses_158)):
    for img in label_158_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第159类的采样=================================================
selected_sampled_losses_159 = []
ms_159 = MeanShift()
ms_159.fit(label_159_loss)

labels_159 = ms_159.labels_  # 第0类中的每个样本对应的簇号

n_clusters_159 = len(np.unique(labels_159))  # 第0类样本有几个簇

need_sampled_cluster_159 = []  # 需要下采样的簇号

cluster_to_num_159 = Counter(labels_159)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_159.keys():
    if cluster_to_num_159[k] > len(labels_159) // n_clusters_159:
        need_sampled_cluster_159.append(k)

need_sampled_losses_159 = [[] for _ in range(len(need_sampled_cluster_159))]

for i in range(len(need_sampled_cluster_159)):
    for j in range(len(labels_159)):
        if labels_159[j] == need_sampled_cluster_159[i]:
            need_sampled_losses_159[i].append(label_159_loss[j][0])

for j in range(len(labels_159)):
    if labels_159[j] not in need_sampled_cluster_159:
        selected_sampled_losses_159.append(label_159_loss[j][0])

for loss in need_sampled_losses_159:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_159.append(lo)
print(len(label_159_loss))
print(len(set(selected_sampled_losses_159)))
for loss in list(set(selected_sampled_losses_159)):
    for img in label_159_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))

# ==================================================针对第160类的采样=================================================

selected_sampled_losses_160 = []

ms_160 = MeanShift()
ms_160.fit(label_160_loss)

labels_160 = ms_160.labels_  # 第0类中的每个样本对应的簇号

n_clusters_160 = len(np.unique(labels_160))  # 第0类样本有几个簇

need_sampled_cluster_160 = []  # 需要下采样的簇号

cluster_to_num_160 = Counter(labels_160)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_160.keys():
    if cluster_to_num_160[k] > len(labels_160) // n_clusters_160:
        need_sampled_cluster_160.append(k)

need_sampled_losses_160 = [[] for _ in range(len(need_sampled_cluster_160))]

for i in range(len(need_sampled_cluster_160)):
    for j in range(len(labels_160)):
        if labels_160[j] == need_sampled_cluster_160[i]:
            need_sampled_losses_160[i].append(label_160_loss[j][0])

for j in range(len(labels_160)):
    if labels_160[j] not in need_sampled_cluster_160:
        selected_sampled_losses_160.append(label_160_loss[j][0])

for loss in need_sampled_losses_160:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_160.append(lo)
print(len(label_160_loss))
print(len(set(selected_sampled_losses_160)))

for loss in list(set(selected_sampled_losses_160)):
    for img in label_160_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第161类的采样=================================================
selected_sampled_losses_161 = []

ms_161 = MeanShift()
ms_161.fit(label_161_loss)

labels_161 = ms_161.labels_  # 第0类中的每个样本对应的簇号

n_clusters_161 = len(np.unique(labels_161))  # 第0类样本有几个簇

need_sampled_cluster_161 = []  # 需要下采样的簇号

cluster_to_num_161 = Counter(labels_161)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_161.keys():
    if cluster_to_num_161[k] > len(labels_161) // n_clusters_161:
        need_sampled_cluster_161.append(k)

need_sampled_losses_161 = [[] for _ in range(len(need_sampled_cluster_161))]

for i in range(len(need_sampled_cluster_161)):
    for j in range(len(labels_161)):
        if labels_161[j] == need_sampled_cluster_161[i]:
            need_sampled_losses_161[i].append(label_161_loss[j][0])

for j in range(len(labels_161)):
    if labels_161[j] not in need_sampled_cluster_161:
        selected_sampled_losses_161.append(label_161_loss[j][0])

for loss in need_sampled_losses_161:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_161.append(lo)
print(len(label_161_loss))
print(len(set(selected_sampled_losses_161)))

for loss in list(set(selected_sampled_losses_161)):
    for img in label_161_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第162类的采样=================================================
selected_sampled_losses_162 = []

ms_162 = MeanShift()
ms_162.fit(label_162_loss)

labels_162 = ms_162.labels_  # 第0类中的每个样本对应的簇号

n_clusters_162 = len(np.unique(labels_162))  # 第0类样本有几个簇

need_sampled_cluster_162 = []  # 需要下采样的簇号

cluster_to_num_162 = Counter(labels_162)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_162.keys():
    if cluster_to_num_162[k] > len(labels_162) // n_clusters_162:
        need_sampled_cluster_162.append(k)

need_sampled_losses_162 = [[] for _ in range(len(need_sampled_cluster_162))]

for i in range(len(need_sampled_cluster_162)):
    for j in range(len(labels_162)):
        if labels_162[j] == need_sampled_cluster_162[i]:
            need_sampled_losses_162[i].append(label_162_loss[j][0])

for j in range(len(labels_162)):
    if labels_162[j] not in need_sampled_cluster_162:
        selected_sampled_losses_162.append(label_162_loss[j][0])

for loss in need_sampled_losses_162:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_162.append(lo)
print(len(label_162_loss))
print(len(set(selected_sampled_losses_162)))

for loss in list(set(selected_sampled_losses_162)):
    for img in label_162_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第163类的采样=================================================
selected_sampled_losses_163 = []

ms_163 = MeanShift()
ms_163.fit(label_163_loss)

labels_163 = ms_163.labels_  # 第0类中的每个样本对应的簇号

n_clusters_163 = len(np.unique(labels_163))  # 第0类样本有几个簇

need_sampled_cluster_163 = []  # 需要下采样的簇号

cluster_to_num_163 = Counter(labels_163)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_163.keys():
    if cluster_to_num_163[k] > len(labels_163) // n_clusters_163:
        need_sampled_cluster_163.append(k)

need_sampled_losses_163 = [[] for _ in range(len(need_sampled_cluster_163))]

for i in range(len(need_sampled_cluster_163)):
    for j in range(len(labels_163)):
        if labels_163[j] == need_sampled_cluster_163[i]:
            need_sampled_losses_163[i].append(label_163_loss[j][0])

for j in range(len(labels_163)):
    if labels_163[j] not in need_sampled_cluster_163:
        selected_sampled_losses_163.append(label_163_loss[j][0])

for loss in need_sampled_losses_163:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_163.append(lo)
print(len(label_163_loss))
print(len(set(selected_sampled_losses_163)))

for loss in list(set(selected_sampled_losses_163)):
    for img in label_163_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第164类的采样=================================================
selected_sampled_losses_164 = []
ms_164 = MeanShift()
ms_164.fit(label_164_loss)

labels_164 = ms_164.labels_  # 第0类中的每个样本对应的簇号

n_clusters_164 = len(np.unique(labels_164))  # 第0类样本有几个簇

need_sampled_cluster_164 = []  # 需要下采样的簇号

cluster_to_num_164 = Counter(labels_164)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_164.keys():
    if cluster_to_num_164[k] > len(labels_164) // n_clusters_164:
        need_sampled_cluster_164.append(k)

need_sampled_losses_164 = [[] for _ in range(len(need_sampled_cluster_164))]

for i in range(len(need_sampled_cluster_164)):
    for j in range(len(labels_164)):
        if labels_164[j] == need_sampled_cluster_164[i]:
            need_sampled_losses_164[i].append(label_164_loss[j][0])

for j in range(len(labels_164)):
    if labels_164[j] not in need_sampled_cluster_164:
        selected_sampled_losses_164.append(label_164_loss[j][0])

for loss in need_sampled_losses_164:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_164.append(lo)
print(len(label_164_loss))
print(len(set(selected_sampled_losses_164)))

for loss in list(set(selected_sampled_losses_164)):
    for img in label_164_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第165类的采样=================================================
selected_sampled_losses_165 = []
ms_165 = MeanShift()
ms_165.fit(label_165_loss)

labels_165 = ms_165.labels_  # 第0类中的每个样本对应的簇号

n_clusters_165 = len(np.unique(labels_165))  # 第0类样本有几个簇

need_sampled_cluster_165 = []  # 需要下采样的簇号

cluster_to_num_165 = Counter(labels_165)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_165.keys():
    if cluster_to_num_165[k] > len(labels_165) // n_clusters_165:
        need_sampled_cluster_165.append(k)

need_sampled_losses_165 = [[] for _ in range(len(need_sampled_cluster_165))]

for i in range(len(need_sampled_cluster_165)):
    for j in range(len(labels_165)):
        if labels_165[j] == need_sampled_cluster_165[i]:
            need_sampled_losses_165[i].append(label_165_loss[j][0])

for j in range(len(labels_165)):
    if labels_165[j] not in need_sampled_cluster_165:
        selected_sampled_losses_165.append(label_165_loss[j][0])

for loss in need_sampled_losses_165:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_165.append(lo)
print(len(label_165_loss))
print(len(set(selected_sampled_losses_165)))

for loss in list(set(selected_sampled_losses_165)):
    for img in label_165_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第166类的采样=================================================
selected_sampled_losses_166 = []
ms_166 = MeanShift()
ms_166.fit(label_166_loss)

labels_166 = ms_166.labels_  # 第0类中的每个样本对应的簇号

n_clusters_166 = len(np.unique(labels_166))  # 第0类样本有几个簇

need_sampled_cluster_166 = []  # 需要下采样的簇号

cluster_to_num_166 = Counter(labels_166)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_166.keys():
    if cluster_to_num_166[k] > len(labels_166) // n_clusters_166:
        need_sampled_cluster_166.append(k)

need_sampled_losses_166 = [[] for _ in range(len(need_sampled_cluster_166))]

for i in range(len(need_sampled_cluster_166)):
    for j in range(len(labels_166)):
        if labels_166[j] == need_sampled_cluster_166[i]:
            need_sampled_losses_166[i].append(label_166_loss[j][0])

for j in range(len(labels_166)):
    if labels_166[j] not in need_sampled_cluster_166:
        selected_sampled_losses_166.append(label_166_loss[j][0])

for loss in need_sampled_losses_166:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_166.append(lo)
print(len(label_166_loss))
print(len(set(selected_sampled_losses_166)))

for loss in list(set(selected_sampled_losses_166)):
    for img in label_166_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第167类的采样=================================================
selected_sampled_losses_167 = []
ms_167 = MeanShift()
ms_167.fit(label_167_loss)

labels_167 = ms_167.labels_  # 第0类中的每个样本对应的簇号

n_clusters_167 = len(np.unique(labels_167))  # 第0类样本有几个簇

need_sampled_cluster_167 = []  # 需要下采样的簇号

cluster_to_num_167 = Counter(labels_167)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_167.keys():
    if cluster_to_num_167[k] > len(labels_167) // n_clusters_167:
        need_sampled_cluster_167.append(k)

need_sampled_losses_167 = [[] for _ in range(len(need_sampled_cluster_167))]

for i in range(len(need_sampled_cluster_167)):
    for j in range(len(labels_167)):
        if labels_167[j] == need_sampled_cluster_167[i]:
            need_sampled_losses_167[i].append(label_167_loss[j][0])

for j in range(len(labels_167)):
    if labels_167[j] not in need_sampled_cluster_167:
        selected_sampled_losses_167.append(label_167_loss[j][0])

for loss in need_sampled_losses_167:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_167.append(lo)
print(len(label_167_loss))
print(len(set(selected_sampled_losses_167)))

for loss in list(set(selected_sampled_losses_167)):
    for img in label_167_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第168类的采样=================================================
selected_sampled_losses_168 = []
ms_168 = MeanShift()
ms_168.fit(label_168_loss)

labels_168 = ms_168.labels_  # 第0类中的每个样本对应的簇号

n_clusters_168 = len(np.unique(labels_168))  # 第0类样本有几个簇

need_sampled_cluster_168 = []  # 需要下采样的簇号

cluster_to_num_168 = Counter(labels_168)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_168.keys():
    if cluster_to_num_168[k] > len(labels_168) // n_clusters_168:
        need_sampled_cluster_168.append(k)

need_sampled_losses_168 = [[] for _ in range(len(need_sampled_cluster_168))]

for i in range(len(need_sampled_cluster_168)):
    for j in range(len(labels_168)):
        if labels_168[j] == need_sampled_cluster_168[i]:
            need_sampled_losses_168[i].append(label_168_loss[j][0])

for j in range(len(labels_168)):
    if labels_168[j] not in need_sampled_cluster_168:
        selected_sampled_losses_168.append(label_168_loss[j][0])

for loss in need_sampled_losses_168:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_168.append(lo)
print(len(label_168_loss))
print(len(set(selected_sampled_losses_168)))

for loss in list(set(selected_sampled_losses_168)):
    for img in label_168_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第169类的采样=================================================
selected_sampled_losses_169 = []
ms_169 = MeanShift()
ms_169.fit(label_169_loss)

labels_169 = ms_169.labels_  # 第0类中的每个样本对应的簇号

n_clusters_169 = len(np.unique(labels_169))  # 第0类样本有几个簇

need_sampled_cluster_169 = []  # 需要下采样的簇号

cluster_to_num_169 = Counter(labels_169)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_169.keys():
    if cluster_to_num_169[k] > len(labels_169) // n_clusters_169:
        need_sampled_cluster_169.append(k)

need_sampled_losses_169 = [[] for _ in range(len(need_sampled_cluster_169))]

for i in range(len(need_sampled_cluster_169)):
    for j in range(len(labels_169)):
        if labels_169[j] == need_sampled_cluster_169[i]:
            need_sampled_losses_169[i].append(label_169_loss[j][0])

for j in range(len(labels_169)):
    if labels_169[j] not in need_sampled_cluster_169:
        selected_sampled_losses_169.append(label_169_loss[j][0])

for loss in need_sampled_losses_169:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_169.append(lo)
print(len(label_169_loss))
print(len(set(selected_sampled_losses_169)))

for loss in list(set(selected_sampled_losses_169)):
    for img in label_169_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第170类的采样=================================================
selected_sampled_losses_170 = []
ms_170 = MeanShift()
ms_170.fit(label_170_loss)

labels_170 = ms_170.labels_  # 第0类中的每个样本对应的簇号

n_clusters_170 = len(np.unique(labels_170))  # 第0类样本有几个簇

need_sampled_cluster_170 = []  # 需要下采样的簇号

cluster_to_num_170 = Counter(labels_170)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_170.keys():
    if cluster_to_num_170[k] > len(labels_170) // n_clusters_170:
        need_sampled_cluster_170.append(k)

need_sampled_losses_170 = [[] for _ in range(len(need_sampled_cluster_170))]

for i in range(len(need_sampled_cluster_170)):
    for j in range(len(labels_170)):
        if labels_170[j] == need_sampled_cluster_170[i]:
            need_sampled_losses_170[i].append(label_170_loss[j][0])

for j in range(len(labels_170)):
    if labels_170[j] not in need_sampled_cluster_170:
        selected_sampled_losses_170.append(label_170_loss[j][0])

for loss in need_sampled_losses_170:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_170.append(lo)
print(len(label_170_loss))
print(len(set(selected_sampled_losses_170)))

for loss in list(set(selected_sampled_losses_170)):
    for img in label_170_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第171类的采样=================================================
selected_sampled_losses_171 = []

ms_171 = MeanShift()
ms_171.fit(label_171_loss)

labels_171 = ms_171.labels_  # 第0类中的每个样本对应的簇号

n_clusters_171 = len(np.unique(labels_171))  # 第0类样本有几个簇

need_sampled_cluster_171 = []  # 需要下采样的簇号

cluster_to_num_171 = Counter(labels_171)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_171.keys():
    if cluster_to_num_171[k] > len(labels_171) // n_clusters_171:
        need_sampled_cluster_171.append(k)

need_sampled_losses_171 = [[] for _ in range(len(need_sampled_cluster_171))]

for i in range(len(need_sampled_cluster_171)):
    for j in range(len(labels_171)):
        if labels_171[j] == need_sampled_cluster_171[i]:
            need_sampled_losses_171[i].append(label_171_loss[j][0])

for j in range(len(labels_171)):
    if labels_171[j] not in need_sampled_cluster_171:
        selected_sampled_losses_171.append(label_171_loss[j][0])

for loss in need_sampled_losses_171:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_171.append(lo)
print(len(label_171_loss))
print(len(set(selected_sampled_losses_171)))

for loss in list(set(selected_sampled_losses_171)):
    for img in label_171_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第172类的采样=================================================
selected_sampled_losses_172 = []
ms_172 = MeanShift()
ms_172.fit(label_172_loss)

labels_172 = ms_172.labels_  # 第0类中的每个样本对应的簇号

n_clusters_172 = len(np.unique(labels_172))  # 第0类样本有几个簇

need_sampled_cluster_172 = []  # 需要下采样的簇号

cluster_to_num_172 = Counter(labels_172)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_172.keys():
    if cluster_to_num_172[k] > len(labels_172) // n_clusters_172:
        need_sampled_cluster_172.append(k)

need_sampled_losses_172 = [[] for _ in range(len(need_sampled_cluster_172))]

for i in range(len(need_sampled_cluster_172)):
    for j in range(len(labels_172)):
        if labels_172[j] == need_sampled_cluster_172[i]:
            need_sampled_losses_172[i].append(label_172_loss[j][0])

for j in range(len(labels_172)):
    if labels_172[j] not in need_sampled_cluster_172:
        selected_sampled_losses_172.append(label_172_loss[j][0])

for loss in need_sampled_losses_172:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_172.append(lo)
print(len(label_172_loss))
print(len(set(selected_sampled_losses_172)))

for loss in list(set(selected_sampled_losses_172)):
    for img in label_172_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第173类的采样=================================================
selected_sampled_losses_173 = []
ms_173 = MeanShift()
ms_173.fit(label_173_loss)

labels_173 = ms_173.labels_  # 第0类中的每个样本对应的簇号

n_clusters_173 = len(np.unique(labels_173))  # 第0类样本有几个簇

need_sampled_cluster_173 = []  # 需要下采样的簇号

cluster_to_num_173 = Counter(labels_173)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_173.keys():
    if cluster_to_num_173[k] > len(labels_173) // n_clusters_173:
        need_sampled_cluster_173.append(k)

need_sampled_losses_173 = [[] for _ in range(len(need_sampled_cluster_173))]

for i in range(len(need_sampled_cluster_173)):
    for j in range(len(labels_173)):
        if labels_173[j] == need_sampled_cluster_173[i]:
            need_sampled_losses_173[i].append(label_173_loss[j][0])

for j in range(len(labels_173)):
    if labels_173[j] not in need_sampled_cluster_173:
        selected_sampled_losses_173.append(label_173_loss[j][0])

for loss in need_sampled_losses_173:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_173.append(lo)
print(len(label_173_loss))
print(len(set(selected_sampled_losses_173)))

for loss in list(set(selected_sampled_losses_173)):
    for img in label_173_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第174类的采样=================================================
selected_sampled_losses_174 = []
ms_174 = MeanShift()
ms_174.fit(label_174_loss)

labels_174 = ms_174.labels_  # 第0类中的每个样本对应的簇号

n_clusters_174 = len(np.unique(labels_174))  # 第0类样本有几个簇

need_sampled_cluster_174 = []  # 需要下采样的簇号

cluster_to_num_174 = Counter(labels_174)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_174.keys():
    if cluster_to_num_174[k] > len(labels_174) // n_clusters_174:
        need_sampled_cluster_174.append(k)

need_sampled_losses_174 = [[] for _ in range(len(need_sampled_cluster_174))]

for i in range(len(need_sampled_cluster_174)):
    for j in range(len(labels_174)):
        if labels_174[j] == need_sampled_cluster_174[i]:
            need_sampled_losses_174[i].append(label_174_loss[j][0])

for j in range(len(labels_174)):
    if labels_174[j] not in need_sampled_cluster_174:
        selected_sampled_losses_174.append(label_174_loss[j][0])

for loss in need_sampled_losses_174:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_174.append(lo)
print(len(label_174_loss))
print(len(set(selected_sampled_losses_174)))

for loss in list(set(selected_sampled_losses_174)):
    for img in label_174_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第175类的采样=================================================
selected_sampled_losses_175 = []
ms_175 = MeanShift()
ms_175.fit(label_175_loss)

labels_175 = ms_175.labels_  # 第0类中的每个样本对应的簇号

n_clusters_175 = len(np.unique(labels_175))  # 第0类样本有几个簇

need_sampled_cluster_175 = []  # 需要下采样的簇号

cluster_to_num_175 = Counter(labels_175)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_175.keys():
    if cluster_to_num_175[k] > len(labels_175) // n_clusters_175:
        need_sampled_cluster_175.append(k)

need_sampled_losses_175 = [[] for _ in range(len(need_sampled_cluster_175))]

for i in range(len(need_sampled_cluster_175)):
    for j in range(len(labels_175)):
        if labels_175[j] == need_sampled_cluster_175[i]:
            need_sampled_losses_175[i].append(label_175_loss[j][0])

for j in range(len(labels_175)):
    if labels_175[j] not in need_sampled_cluster_175:
        selected_sampled_losses_175.append(label_175_loss[j][0])

for loss in need_sampled_losses_175:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_175.append(lo)
print(len(label_175_loss))
print(len(set(selected_sampled_losses_175)))

for loss in list(set(selected_sampled_losses_175)):
    for img in label_175_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第176类的采样=================================================
selected_sampled_losses_176 = []
ms_176 = MeanShift()
ms_176.fit(label_176_loss)

labels_176 = ms_176.labels_  # 第0类中的每个样本对应的簇号

n_clusters_176 = len(np.unique(labels_176))  # 第0类样本有几个簇

need_sampled_cluster_176 = []  # 需要下采样的簇号

cluster_to_num_176 = Counter(labels_176)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_176.keys():
    if cluster_to_num_176[k] > len(labels_176) // n_clusters_176:
        need_sampled_cluster_176.append(k)

need_sampled_losses_176 = [[] for _ in range(len(need_sampled_cluster_176))]

for i in range(len(need_sampled_cluster_176)):
    for j in range(len(labels_176)):
        if labels_176[j] == need_sampled_cluster_176[i]:
            need_sampled_losses_176[i].append(label_176_loss[j][0])

for j in range(len(labels_176)):
    if labels_176[j] not in need_sampled_cluster_176:
        selected_sampled_losses_176.append(label_176_loss[j][0])

for loss in need_sampled_losses_176:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_176.append(lo)
print(len(label_176_loss))
print(len(set(selected_sampled_losses_176)))

for loss in list(set(selected_sampled_losses_176)):
    for img in label_176_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第177类的采样=================================================
selected_sampled_losses_177 = []
ms_177 = MeanShift()
ms_177.fit(label_177_loss)

labels_177 = ms_177.labels_  # 第0类中的每个样本对应的簇号

n_clusters_177 = len(np.unique(labels_177))  # 第0类样本有几个簇

need_sampled_cluster_177 = []  # 需要下采样的簇号

cluster_to_num_177 = Counter(labels_177)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_177.keys():
    if cluster_to_num_177[k] > len(labels_177) // n_clusters_177:
        need_sampled_cluster_177.append(k)

need_sampled_losses_177 = [[] for _ in range(len(need_sampled_cluster_177))]

for i in range(len(need_sampled_cluster_177)):
    for j in range(len(labels_177)):
        if labels_177[j] == need_sampled_cluster_177[i]:
            need_sampled_losses_177[i].append(label_177_loss[j][0])

for j in range(len(labels_177)):
    if labels_177[j] not in need_sampled_cluster_177:
        selected_sampled_losses_177.append(label_177_loss[j][0])

for loss in need_sampled_losses_177:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_177.append(lo)
print(len(label_177_loss))
print(len(set(selected_sampled_losses_177)))

for loss in list(set(selected_sampled_losses_177)):
    for img in label_177_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第178类的采样=================================================
selected_sampled_losses_178 = []
ms_178 = MeanShift()
ms_178.fit(label_178_loss)

labels_178 = ms_178.labels_  # 第0类中的每个样本对应的簇号

n_clusters_178 = len(np.unique(labels_178))  # 第0类样本有几个簇

need_sampled_cluster_178 = []  # 需要下采样的簇号

cluster_to_num_178 = Counter(labels_178)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_178.keys():
    if cluster_to_num_178[k] > len(labels_178) // n_clusters_178:
        need_sampled_cluster_178.append(k)

need_sampled_losses_178 = [[] for _ in range(len(need_sampled_cluster_178))]

for i in range(len(need_sampled_cluster_178)):
    for j in range(len(labels_178)):
        if labels_178[j] == need_sampled_cluster_178[i]:
            need_sampled_losses_178[i].append(label_178_loss[j][0])

for j in range(len(labels_178)):
    if labels_178[j] not in need_sampled_cluster_178:
        selected_sampled_losses_178.append(label_178_loss[j][0])

for loss in need_sampled_losses_178:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_178.append(lo)
print(len(label_178_loss))
print(len(set(selected_sampled_losses_178)))

for loss in list(set(selected_sampled_losses_178)):
    for img in label_178_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第179类的采样=================================================
selected_sampled_losses_179 = []
ms_179 = MeanShift()
ms_179.fit(label_179_loss)

labels_179 = ms_179.labels_  # 第0类中的每个样本对应的簇号

n_clusters_179 = len(np.unique(labels_179))  # 第0类样本有几个簇

need_sampled_cluster_179 = []  # 需要下采样的簇号

cluster_to_num_179 = Counter(labels_179)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_179.keys():
    if cluster_to_num_179[k] > len(labels_179) // n_clusters_179:
        need_sampled_cluster_179.append(k)

need_sampled_losses_179 = [[] for _ in range(len(need_sampled_cluster_179))]

for i in range(len(need_sampled_cluster_179)):
    for j in range(len(labels_179)):
        if labels_179[j] == need_sampled_cluster_179[i]:
            need_sampled_losses_179[i].append(label_179_loss[j][0])

for j in range(len(labels_179)):
    if labels_179[j] not in need_sampled_cluster_179:
        selected_sampled_losses_179.append(label_179_loss[j][0])

for loss in need_sampled_losses_179:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_179.append(lo)
print(len(label_179_loss))
print(len(set(selected_sampled_losses_179)))
for loss in list(set(selected_sampled_losses_179)):
    for img in label_179_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))

# ==================================================针对第180类的采样=================================================

selected_sampled_losses_180 = []

ms_180 = MeanShift()
ms_180.fit(label_180_loss)

labels_180 = ms_180.labels_  # 第_180类中的每个样本对应的簇号

n_clusters_180 = len(np.unique(labels_180))  # 第_180类样本有几个簇

need_sampled_cluster_180 = []  # 需要下采样的簇号

cluster_to_num_180 = Counter(labels_180)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_180.keys():
    if cluster_to_num_180[k] > len(labels_180) // n_clusters_180:
        need_sampled_cluster_180.append(k)

need_sampled_losses_180 = [[] for _ in range(len(need_sampled_cluster_180))]

for i in range(len(need_sampled_cluster_180)):
    for j in range(len(labels_180)):
        if labels_180[j] == need_sampled_cluster_180[i]:
            need_sampled_losses_180[i].append(label_180_loss[j][0])

for j in range(len(labels_180)):
    if labels_180[j] not in need_sampled_cluster_180:
        selected_sampled_losses_180.append(label_180_loss[j][0])

for loss in need_sampled_losses_180:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_180.append(lo)
print(len(label_180_loss))
print(len(set(selected_sampled_losses_180)))

for loss in list(set(selected_sampled_losses_180)):
    for img in label_180_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第181类的采样=================================================
selected_sampled_losses_181 = []

ms_181 = MeanShift()
ms_181.fit(label_181_loss)

labels_181 = ms_181.labels_  # 第0类中的每个样本对应的簇号

n_clusters_181 = len(np.unique(labels_181))  # 第0类样本有几个簇

need_sampled_cluster_181 = []  # 需要下采样的簇号

cluster_to_num_181 = Counter(labels_181)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_181.keys():
    if cluster_to_num_181[k] > len(labels_181) // n_clusters_181:
        need_sampled_cluster_181.append(k)

need_sampled_losses_181 = [[] for _ in range(len(need_sampled_cluster_181))]

for i in range(len(need_sampled_cluster_181)):
    for j in range(len(labels_181)):
        if labels_181[j] == need_sampled_cluster_181[i]:
            need_sampled_losses_181[i].append(label_181_loss[j][0])

for j in range(len(labels_181)):
    if labels_181[j] not in need_sampled_cluster_181:
        selected_sampled_losses_181.append(label_181_loss[j][0])

for loss in need_sampled_losses_181:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_181.append(lo)
print(len(label_181_loss))
print(len(set(selected_sampled_losses_181)))

for loss in list(set(selected_sampled_losses_181)):
    for img in label_181_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第182类的采样=================================================
selected_sampled_losses_182 = []

ms_182 = MeanShift()
ms_182.fit(label_182_loss)

labels_182 = ms_182.labels_  # 第0类中的每个样本对应的簇号

n_clusters_182 = len(np.unique(labels_182))  # 第0类样本有几个簇

need_sampled_cluster_182 = []  # 需要下采样的簇号

cluster_to_num_182 = Counter(labels_182)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_182.keys():
    if cluster_to_num_182[k] > len(labels_182) // n_clusters_182:
        need_sampled_cluster_182.append(k)

need_sampled_losses_182 = [[] for _ in range(len(need_sampled_cluster_182))]

for i in range(len(need_sampled_cluster_182)):
    for j in range(len(labels_182)):
        if labels_182[j] == need_sampled_cluster_182[i]:
            need_sampled_losses_182[i].append(label_182_loss[j][0])

for j in range(len(labels_182)):
    if labels_182[j] not in need_sampled_cluster_182:
        selected_sampled_losses_182.append(label_182_loss[j][0])

for loss in need_sampled_losses_182:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_182.append(lo)
print(len(label_182_loss))
print(len(set(selected_sampled_losses_182)))

for loss in list(set(selected_sampled_losses_182)):
    for img in label_182_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第183类的采样=================================================
selected_sampled_losses_183 = []

ms_183 = MeanShift()
ms_183.fit(label_183_loss)

labels_183 = ms_183.labels_  # 第0类中的每个样本对应的簇号

n_clusters_183 = len(np.unique(labels_183))  # 第0类样本有几个簇

need_sampled_cluster_183 = []  # 需要下采样的簇号

cluster_to_num_183 = Counter(labels_183)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_183.keys():
    if cluster_to_num_183[k] > len(labels_183) // n_clusters_183:
        need_sampled_cluster_183.append(k)

need_sampled_losses_183 = [[] for _ in range(len(need_sampled_cluster_183))]

for i in range(len(need_sampled_cluster_183)):
    for j in range(len(labels_183)):
        if labels_183[j] == need_sampled_cluster_183[i]:
            need_sampled_losses_183[i].append(label_183_loss[j][0])

for j in range(len(labels_183)):
    if labels_183[j] not in need_sampled_cluster_183:
        selected_sampled_losses_183.append(label_183_loss[j][0])

for loss in need_sampled_losses_183:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_183.append(lo)
print(len(label_183_loss))
print(len(set(selected_sampled_losses_183)))

for loss in list(set(selected_sampled_losses_183)):
    for img in label_183_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第184类的采样=================================================
selected_sampled_losses_184 = []
ms_184 = MeanShift()
ms_184.fit(label_184_loss)

labels_184 = ms_184.labels_  # 第0类中的每个样本对应的簇号

n_clusters_184 = len(np.unique(labels_184))  # 第0类样本有几个簇

need_sampled_cluster_184 = []  # 需要下采样的簇号

cluster_to_num_184 = Counter(labels_184)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_184.keys():
    if cluster_to_num_184[k] > len(labels_184) // n_clusters_184:
        need_sampled_cluster_184.append(k)

need_sampled_losses_184 = [[] for _ in range(len(need_sampled_cluster_184))]

for i in range(len(need_sampled_cluster_184)):
    for j in range(len(labels_184)):
        if labels_184[j] == need_sampled_cluster_184[i]:
            need_sampled_losses_184[i].append(label_184_loss[j][0])

for j in range(len(labels_184)):
    if labels_184[j] not in need_sampled_cluster_184:
        selected_sampled_losses_184.append(label_184_loss[j][0])

for loss in need_sampled_losses_184:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_184.append(lo)
print(len(label_184_loss))
print(len(set(selected_sampled_losses_184)))

for loss in list(set(selected_sampled_losses_184)):
    for img in label_184_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第185类的采样=================================================
selected_sampled_losses_185 = []
ms_185 = MeanShift()
ms_185.fit(label_185_loss)

labels_185 = ms_185.labels_  # 第0类中的每个样本对应的簇号

n_clusters_185 = len(np.unique(labels_185))  # 第0类样本有几个簇

need_sampled_cluster_185 = []  # 需要下采样的簇号

cluster_to_num_185 = Counter(labels_185)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_185.keys():
    if cluster_to_num_185[k] > len(labels_185) // n_clusters_185:
        need_sampled_cluster_185.append(k)

need_sampled_losses_185 = [[] for _ in range(len(need_sampled_cluster_185))]

for i in range(len(need_sampled_cluster_185)):
    for j in range(len(labels_185)):
        if labels_185[j] == need_sampled_cluster_185[i]:
            need_sampled_losses_185[i].append(label_185_loss[j][0])

for j in range(len(labels_185)):
    if labels_185[j] not in need_sampled_cluster_185:
        selected_sampled_losses_185.append(label_185_loss[j][0])

for loss in need_sampled_losses_185:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_185.append(lo)
print(len(label_185_loss))
print(len(set(selected_sampled_losses_185)))

for loss in list(set(selected_sampled_losses_185)):
    for img in label_185_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第186类的采样=================================================
selected_sampled_losses_186 = []
ms_186 = MeanShift()
ms_186.fit(label_186_loss)

labels_186 = ms_186.labels_  # 第0类中的每个样本对应的簇号

n_clusters_186 = len(np.unique(labels_186))  # 第0类样本有几个簇

need_sampled_cluster_186 = []  # 需要下采样的簇号

cluster_to_num_186 = Counter(labels_186)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_186.keys():
    if cluster_to_num_186[k] > len(labels_186) // n_clusters_186:
        need_sampled_cluster_186.append(k)

need_sampled_losses_186 = [[] for _ in range(len(need_sampled_cluster_186))]

for i in range(len(need_sampled_cluster_186)):
    for j in range(len(labels_186)):
        if labels_186[j] == need_sampled_cluster_186[i]:
            need_sampled_losses_186[i].append(label_186_loss[j][0])

for j in range(len(labels_186)):
    if labels_186[j] not in need_sampled_cluster_186:
        selected_sampled_losses_186.append(label_186_loss[j][0])

for loss in need_sampled_losses_186:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_186.append(lo)
print(len(label_186_loss))
print(len(set(selected_sampled_losses_186)))

for loss in list(set(selected_sampled_losses_186)):
    for img in label_186_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第187类的采样=================================================
selected_sampled_losses_187 = []
ms_187 = MeanShift()
ms_187.fit(label_187_loss)

labels_187 = ms_187.labels_  # 第0类中的每个样本对应的簇号

n_clusters_187 = len(np.unique(labels_187))  # 第0类样本有几个簇

need_sampled_cluster_187 = []  # 需要下采样的簇号

cluster_to_num_187 = Counter(labels_187)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_187.keys():
    if cluster_to_num_187[k] > len(labels_187) // n_clusters_187:
        need_sampled_cluster_187.append(k)

need_sampled_losses_187 = [[] for _ in range(len(need_sampled_cluster_187))]

for i in range(len(need_sampled_cluster_187)):
    for j in range(len(labels_187)):
        if labels_187[j] == need_sampled_cluster_187[i]:
            need_sampled_losses_187[i].append(label_187_loss[j][0])

for j in range(len(labels_187)):
    if labels_187[j] not in need_sampled_cluster_187:
        selected_sampled_losses_187.append(label_187_loss[j][0])

for loss in need_sampled_losses_187:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_187.append(lo)
print(len(label_187_loss))
print(len(set(selected_sampled_losses_187)))

for loss in list(set(selected_sampled_losses_187)):
    for img in label_187_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第188类的采样=================================================
selected_sampled_losses_188 = []
ms_188 = MeanShift()
ms_188.fit(label_188_loss)

labels_188 = ms_188.labels_  # 第0类中的每个样本对应的簇号

n_clusters_188 = len(np.unique(labels_188))  # 第0类样本有几个簇

need_sampled_cluster_188 = []  # 需要下采样的簇号

cluster_to_num_188 = Counter(labels_188)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_188.keys():
    if cluster_to_num_188[k] > len(labels_188) // n_clusters_188:
        need_sampled_cluster_188.append(k)

need_sampled_losses_188 = [[] for _ in range(len(need_sampled_cluster_188))]

for i in range(len(need_sampled_cluster_188)):
    for j in range(len(labels_188)):
        if labels_188[j] == need_sampled_cluster_188[i]:
            need_sampled_losses_188[i].append(label_188_loss[j][0])

for j in range(len(labels_188)):
    if labels_188[j] not in need_sampled_cluster_188:
        selected_sampled_losses_188.append(label_188_loss[j][0])

for loss in need_sampled_losses_188:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_188.append(lo)
print(len(label_188_loss))
print(len(set(selected_sampled_losses_188)))

for loss in list(set(selected_sampled_losses_188)):
    for img in label_188_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第189类的采样=================================================
selected_sampled_losses_189 = []
ms_189 = MeanShift()
ms_189.fit(label_189_loss)

labels_189 = ms_189.labels_  # 第0类中的每个样本对应的簇号

n_clusters_189 = len(np.unique(labels_189))  # 第0类样本有几个簇

need_sampled_cluster_189 = []  # 需要下采样的簇号

cluster_to_num_189 = Counter(labels_189)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_189.keys():
    if cluster_to_num_189[k] > len(labels_189) // n_clusters_189:
        need_sampled_cluster_189.append(k)

need_sampled_losses_189 = [[] for _ in range(len(need_sampled_cluster_189))]

for i in range(len(need_sampled_cluster_189)):
    for j in range(len(labels_189)):
        if labels_189[j] == need_sampled_cluster_189[i]:
            need_sampled_losses_189[i].append(label_189_loss[j][0])

for j in range(len(labels_189)):
    if labels_189[j] not in need_sampled_cluster_189:
        selected_sampled_losses_189.append(label_189_loss[j][0])

for loss in need_sampled_losses_189:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_189.append(lo)
print(len(label_189_loss))
print(len(set(selected_sampled_losses_189)))

for loss in list(set(selected_sampled_losses_189)):
    for img in label_189_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第190类的采样=================================================
selected_sampled_losses_190 = []
ms_190 = MeanShift()
ms_190.fit(label_190_loss)

labels_190 = ms_190.labels_  # 第0类中的每个样本对应的簇号

n_clusters_190 = len(np.unique(labels_190))  # 第0类样本有几个簇

need_sampled_cluster_190 = []  # 需要下采样的簇号

cluster_to_num_190 = Counter(labels_190)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_190.keys():
    if cluster_to_num_190[k] > len(labels_190) // n_clusters_190:
        need_sampled_cluster_190.append(k)

need_sampled_losses_190 = [[] for _ in range(len(need_sampled_cluster_190))]

for i in range(len(need_sampled_cluster_190)):
    for j in range(len(labels_190)):
        if labels_190[j] == need_sampled_cluster_190[i]:
            need_sampled_losses_190[i].append(label_190_loss[j][0])

for j in range(len(labels_190)):
    if labels_190[j] not in need_sampled_cluster_190:
        selected_sampled_losses_190.append(label_190_loss[j][0])

for loss in need_sampled_losses_190:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_190.append(lo)
print(len(label_190_loss))
print(len(set(selected_sampled_losses_190)))

for loss in list(set(selected_sampled_losses_190)):
    for img in label_190_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第191类的采样=================================================
selected_sampled_losses_191 = []

ms_191 = MeanShift()
ms_191.fit(label_191_loss)

labels_191 = ms_191.labels_  # 第0类中的每个样本对应的簇号

n_clusters_191 = len(np.unique(labels_191))  # 第0类样本有几个簇

need_sampled_cluster_191 = []  # 需要下采样的簇号

cluster_to_num_191 = Counter(labels_191)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_191.keys():
    if cluster_to_num_191[k] > len(labels_191) // n_clusters_191:
        need_sampled_cluster_191.append(k)

need_sampled_losses_191 = [[] for _ in range(len(need_sampled_cluster_191))]

for i in range(len(need_sampled_cluster_191)):
    for j in range(len(labels_191)):
        if labels_191[j] == need_sampled_cluster_191[i]:
            need_sampled_losses_191[i].append(label_191_loss[j][0])

for j in range(len(labels_191)):
    if labels_191[j] not in need_sampled_cluster_191:
        selected_sampled_losses_191.append(label_191_loss[j][0])

for loss in need_sampled_losses_191:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_191.append(lo)
print(len(label_191_loss))
print(len(set(selected_sampled_losses_191)))

for loss in list(set(selected_sampled_losses_191)):
    for img in label_191_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第192类的采样=================================================
selected_sampled_losses_192 = []
ms_192 = MeanShift()
ms_192.fit(label_192_loss)

labels_192 = ms_192.labels_  # 第0类中的每个样本对应的簇号

n_clusters_192 = len(np.unique(labels_192))  # 第0类样本有几个簇

need_sampled_cluster_192 = []  # 需要下采样的簇号

cluster_to_num_192 = Counter(labels_192)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_192.keys():
    if cluster_to_num_192[k] > len(labels_192) // n_clusters_192:
        need_sampled_cluster_192.append(k)

need_sampled_losses_192 = [[] for _ in range(len(need_sampled_cluster_192))]

for i in range(len(need_sampled_cluster_192)):
    for j in range(len(labels_192)):
        if labels_192[j] == need_sampled_cluster_192[i]:
            need_sampled_losses_192[i].append(label_192_loss[j][0])

for j in range(len(labels_192)):
    if labels_192[j] not in need_sampled_cluster_192:
        selected_sampled_losses_192.append(label_192_loss[j][0])

for loss in need_sampled_losses_192:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_192.append(lo)
print(len(label_192_loss))
print(len(set(selected_sampled_losses_192)))

for loss in list(set(selected_sampled_losses_192)):
    for img in label_192_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第193类的采样=================================================
selected_sampled_losses_193 = []
ms_193 = MeanShift()
ms_193.fit(label_193_loss)

labels_193 = ms_193.labels_  # 第0类中的每个样本对应的簇号

n_clusters_193 = len(np.unique(labels_193))  # 第0类样本有几个簇

need_sampled_cluster_193 = []  # 需要下采样的簇号

cluster_to_num_193 = Counter(labels_193)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_193.keys():
    if cluster_to_num_193[k] > len(labels_193) // n_clusters_193:
        need_sampled_cluster_193.append(k)

need_sampled_losses_193 = [[] for _ in range(len(need_sampled_cluster_193))]

for i in range(len(need_sampled_cluster_193)):
    for j in range(len(labels_193)):
        if labels_193[j] == need_sampled_cluster_193[i]:
            need_sampled_losses_193[i].append(label_193_loss[j][0])

for j in range(len(labels_193)):
    if labels_193[j] not in need_sampled_cluster_193:
        selected_sampled_losses_193.append(label_193_loss[j][0])

for loss in need_sampled_losses_193:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_193.append(lo)
print(len(label_193_loss))
print(len(set(selected_sampled_losses_193)))

for loss in list(set(selected_sampled_losses_193)):
    for img in label_193_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第194类的采样=================================================
selected_sampled_losses_194 = []
ms_194 = MeanShift()
ms_194.fit(label_194_loss)

labels_194 = ms_194.labels_  # 第0类中的每个样本对应的簇号

n_clusters_194 = len(np.unique(labels_194))  # 第0类样本有几个簇

need_sampled_cluster_194 = []  # 需要下采样的簇号

cluster_to_num_194 = Counter(labels_194)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_194.keys():
    if cluster_to_num_194[k] > len(labels_194) // n_clusters_194:
        need_sampled_cluster_194.append(k)

need_sampled_losses_194 = [[] for _ in range(len(need_sampled_cluster_194))]

for i in range(len(need_sampled_cluster_194)):
    for j in range(len(labels_194)):
        if labels_194[j] == need_sampled_cluster_194[i]:
            need_sampled_losses_194[i].append(label_194_loss[j][0])

for j in range(len(labels_194)):
    if labels_194[j] not in need_sampled_cluster_194:
        selected_sampled_losses_194.append(label_194_loss[j][0])

for loss in need_sampled_losses_194:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_194.append(lo)
print(len(label_194_loss))
print(len(set(selected_sampled_losses_194)))

for loss in list(set(selected_sampled_losses_194)):
    for img in label_194_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第195类的采样=================================================
selected_sampled_losses_195 = []
ms_195 = MeanShift()
ms_195.fit(label_195_loss)

labels_195 = ms_195.labels_  # 第0类中的每个样本对应的簇号

n_clusters_195 = len(np.unique(labels_195))  # 第0类样本有几个簇

need_sampled_cluster_195 = []  # 需要下采样的簇号

cluster_to_num_195 = Counter(labels_195)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_195.keys():
    if cluster_to_num_195[k] > len(labels_195) // n_clusters_195:
        need_sampled_cluster_195.append(k)

need_sampled_losses_195 = [[] for _ in range(len(need_sampled_cluster_195))]

for i in range(len(need_sampled_cluster_195)):
    for j in range(len(labels_195)):
        if labels_195[j] == need_sampled_cluster_195[i]:
            need_sampled_losses_195[i].append(label_195_loss[j][0])

for j in range(len(labels_195)):
    if labels_195[j] not in need_sampled_cluster_195:
        selected_sampled_losses_195.append(label_195_loss[j][0])

for loss in need_sampled_losses_195:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_195.append(lo)
print(len(label_195_loss))
print(len(set(selected_sampled_losses_195)))

for loss in list(set(selected_sampled_losses_195)):
    for img in label_195_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第196类的采样=================================================
selected_sampled_losses_196 = []
ms_196 = MeanShift()
ms_196.fit(label_196_loss)

labels_196 = ms_196.labels_  # 第0类中的每个样本对应的簇号

n_clusters_196 = len(np.unique(labels_196))  # 第0类样本有几个簇

need_sampled_cluster_196 = []  # 需要下采样的簇号

cluster_to_num_196 = Counter(labels_196)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_196.keys():
    if cluster_to_num_196[k] > len(labels_196) // n_clusters_196:
        need_sampled_cluster_196.append(k)

need_sampled_losses_196 = [[] for _ in range(len(need_sampled_cluster_196))]

for i in range(len(need_sampled_cluster_196)):
    for j in range(len(labels_196)):
        if labels_196[j] == need_sampled_cluster_196[i]:
            need_sampled_losses_196[i].append(label_196_loss[j][0])

for j in range(len(labels_196)):
    if labels_196[j] not in need_sampled_cluster_196:
        selected_sampled_losses_196.append(label_196_loss[j][0])

for loss in need_sampled_losses_196:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_196.append(lo)
print(len(label_196_loss))
print(len(set(selected_sampled_losses_196)))

for loss in list(set(selected_sampled_losses_196)):
    for img in label_196_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第197类的采样=================================================
selected_sampled_losses_197 = []
ms_197 = MeanShift()
ms_197.fit(label_197_loss)

labels_197 = ms_197.labels_  # 第0类中的每个样本对应的簇号

n_clusters_197 = len(np.unique(labels_197))  # 第0类样本有几个簇

need_sampled_cluster_197 = []  # 需要下采样的簇号

cluster_to_num_197 = Counter(labels_197)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_197.keys():
    if cluster_to_num_197[k] > len(labels_197) // n_clusters_197:
        need_sampled_cluster_197.append(k)

need_sampled_losses_197 = [[] for _ in range(len(need_sampled_cluster_197))]

for i in range(len(need_sampled_cluster_197)):
    for j in range(len(labels_197)):
        if labels_197[j] == need_sampled_cluster_197[i]:
            need_sampled_losses_197[i].append(label_197_loss[j][0])

for j in range(len(labels_197)):
    if labels_197[j] not in need_sampled_cluster_197:
        selected_sampled_losses_197.append(label_197_loss[j][0])

for loss in need_sampled_losses_197:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_197.append(lo)
print(len(label_197_loss))
print(len(set(selected_sampled_losses_197)))

for loss in list(set(selected_sampled_losses_197)):
    for img in label_197_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第198类的采样=================================================
selected_sampled_losses_198 = []
ms_198 = MeanShift()
ms_198.fit(label_198_loss)

labels_198 = ms_198.labels_  # 第0类中的每个样本对应的簇号

n_clusters_198 = len(np.unique(labels_198))  # 第0类样本有几个簇

need_sampled_cluster_198 = []  # 需要下采样的簇号

cluster_to_num_198 = Counter(labels_198)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_198.keys():
    if cluster_to_num_198[k] > len(labels_198) // n_clusters_198:
        need_sampled_cluster_198.append(k)

need_sampled_losses_198 = [[] for _ in range(len(need_sampled_cluster_198))]

for i in range(len(need_sampled_cluster_198)):
    for j in range(len(labels_198)):
        if labels_198[j] == need_sampled_cluster_198[i]:
            need_sampled_losses_198[i].append(label_198_loss[j][0])

for j in range(len(labels_198)):
    if labels_198[j] not in need_sampled_cluster_198:
        selected_sampled_losses_198.append(label_198_loss[j][0])

for loss in need_sampled_losses_198:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_198.append(lo)
print(len(label_198_loss))
print(len(set(selected_sampled_losses_198)))

for loss in list(set(selected_sampled_losses_198)):
    for img in label_198_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第199类的采样=================================================
selected_sampled_losses_199 = []
ms_199 = MeanShift()
ms_199.fit(label_199_loss)

labels_199 = ms_199.labels_  # 第0类中的每个样本对应的簇号

n_clusters_199 = len(np.unique(labels_199))  # 第0类样本有几个簇

need_sampled_cluster_199 = []  # 需要下采样的簇号

cluster_to_num_199 = Counter(labels_199)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_199.keys():
    if cluster_to_num_199[k] > len(labels_199) // n_clusters_199:
        need_sampled_cluster_199.append(k)

need_sampled_losses_199 = [[] for _ in range(len(need_sampled_cluster_199))]

for i in range(len(need_sampled_cluster_199)):
    for j in range(len(labels_199)):
        if labels_199[j] == need_sampled_cluster_199[i]:
            need_sampled_losses_199[i].append(label_199_loss[j][0])

for j in range(len(labels_199)):
    if labels_199[j] not in need_sampled_cluster_199:
        selected_sampled_losses_199.append(label_199_loss[j][0])

for loss in need_sampled_losses_199:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_199.append(lo)
print(len(label_199_loss))
print(len(set(selected_sampled_losses_199)))
for loss in list(set(selected_sampled_losses_199)):
    for img in label_199_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))

# ==================================================针对第200类的采样=================================================

selected_sampled_losses_200 = []

ms_200 = MeanShift()
ms_200.fit(label_200_loss)

labels_200 = ms_200.labels_  # 第0类中的每个样本对应的簇号

n_clusters_200 = len(np.unique(labels_200))  # 第0类样本有几个簇

need_sampled_cluster_200 = []  # 需要下采样的簇号

cluster_to_num_200 = Counter(labels_200)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_200.keys():
    if cluster_to_num_200[k] > len(labels_200) // n_clusters_200:
        need_sampled_cluster_200.append(k)

need_sampled_losses_200 = [[] for _ in range(len(need_sampled_cluster_200))]

for i in range(len(need_sampled_cluster_200)):
    for j in range(len(labels_200)):
        if labels_200[j] == need_sampled_cluster_200[i]:
            need_sampled_losses_200[i].append(label_200_loss[j][0])

for j in range(len(labels_200)):
    if labels_200[j] not in need_sampled_cluster_200:
        selected_sampled_losses_200.append(label_200_loss[j][0])

for loss in need_sampled_losses_200:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_200.append(lo)
print(len(label_200_loss))
print(len(set(selected_sampled_losses_200)))

for loss in list(set(selected_sampled_losses_200)):
    for img in label_200_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第201类的采样=================================================
selected_sampled_losses_201 = []

ms_201 = MeanShift()
ms_201.fit(label_201_loss)

labels_201 = ms_201.labels_  # 第0类中的每个样本对应的簇号

n_clusters_201 = len(np.unique(labels_201))  # 第0类样本有几个簇

need_sampled_cluster_201 = []  # 需要下采样的簇号

cluster_to_num_201 = Counter(labels_201)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_201.keys():
    if cluster_to_num_201[k] > len(labels_201) // n_clusters_201:
        need_sampled_cluster_201.append(k)

need_sampled_losses_201 = [[] for _ in range(len(need_sampled_cluster_201))]

for i in range(len(need_sampled_cluster_201)):
    for j in range(len(labels_201)):
        if labels_201[j] == need_sampled_cluster_201[i]:
            need_sampled_losses_201[i].append(label_201_loss[j][0])

for j in range(len(labels_201)):
    if labels_201[j] not in need_sampled_cluster_201:
        selected_sampled_losses_201.append(label_201_loss[j][0])

for loss in need_sampled_losses_201:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_201.append(lo)
print(len(label_201_loss))
print(len(set(selected_sampled_losses_201)))

for loss in list(set(selected_sampled_losses_201)):
    for img in label_201_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第202类的采样=================================================
selected_sampled_losses_202 = []

ms_202 = MeanShift()
ms_202.fit(label_202_loss)

labels_202 = ms_202.labels_  # 第0类中的每个样本对应的簇号

n_clusters_202 = len(np.unique(labels_202))  # 第0类样本有几个簇

need_sampled_cluster_202 = []  # 需要下采样的簇号

cluster_to_num_202 = Counter(labels_202)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_202.keys():
    if cluster_to_num_202[k] > len(labels_202) // n_clusters_202:
        need_sampled_cluster_202.append(k)

need_sampled_losses_202 = [[] for _ in range(len(need_sampled_cluster_202))]

for i in range(len(need_sampled_cluster_202)):
    for j in range(len(labels_202)):
        if labels_202[j] == need_sampled_cluster_202[i]:
            need_sampled_losses_202[i].append(label_202_loss[j][0])

for j in range(len(labels_202)):
    if labels_202[j] not in need_sampled_cluster_202:
        selected_sampled_losses_202.append(label_202_loss[j][0])

for loss in need_sampled_losses_202:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_202.append(lo)
print(len(label_202_loss))
print(len(set(selected_sampled_losses_202)))

for loss in list(set(selected_sampled_losses_202)):
    for img in label_202_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第203类的采样=================================================
selected_sampled_losses_203 = []

ms_203 = MeanShift()
ms_203.fit(label_203_loss)

labels_203 = ms_203.labels_  # 第0类中的每个样本对应的簇号

n_clusters_203 = len(np.unique(labels_203))  # 第0类样本有几个簇

need_sampled_cluster_203 = []  # 需要下采样的簇号

cluster_to_num_203 = Counter(labels_203)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_203.keys():
    if cluster_to_num_203[k] > len(labels_203) // n_clusters_203:
        need_sampled_cluster_203.append(k)

need_sampled_losses_203 = [[] for _ in range(len(need_sampled_cluster_203))]

for i in range(len(need_sampled_cluster_203)):
    for j in range(len(labels_203)):
        if labels_203[j] == need_sampled_cluster_203[i]:
            need_sampled_losses_203[i].append(label_203_loss[j][0])

for j in range(len(labels_203)):
    if labels_203[j] not in need_sampled_cluster_203:
        selected_sampled_losses_203.append(label_203_loss[j][0])

for loss in need_sampled_losses_203:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_203.append(lo)
print(len(label_203_loss))
print(len(set(selected_sampled_losses_203)))

for loss in list(set(selected_sampled_losses_203)):
    for img in label_203_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第204类的采样=================================================
selected_sampled_losses_204 = []
ms_204 = MeanShift()
ms_204.fit(label_204_loss)

labels_204 = ms_204.labels_  # 第0类中的每个样本对应的簇号

n_clusters_204 = len(np.unique(labels_204))  # 第0类样本有几个簇

need_sampled_cluster_204 = []  # 需要下采样的簇号

cluster_to_num_204 = Counter(labels_204)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_204.keys():
    if cluster_to_num_204[k] > len(labels_204) // n_clusters_204:
        need_sampled_cluster_204.append(k)

need_sampled_losses_204 = [[] for _ in range(len(need_sampled_cluster_204))]

for i in range(len(need_sampled_cluster_204)):
    for j in range(len(labels_204)):
        if labels_204[j] == need_sampled_cluster_204[i]:
            need_sampled_losses_204[i].append(label_204_loss[j][0])

for j in range(len(labels_204)):
    if labels_204[j] not in need_sampled_cluster_204:
        selected_sampled_losses_204.append(label_204_loss[j][0])

for loss in need_sampled_losses_204:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_204.append(lo)
print(len(label_204_loss))
print(len(set(selected_sampled_losses_204)))

for loss in list(set(selected_sampled_losses_204)):
    for img in label_204_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第205类的采样=================================================
selected_sampled_losses_205 = []
ms_205 = MeanShift()
ms_205.fit(label_205_loss)

labels_205 = ms_205.labels_  # 第0类中的每个样本对应的簇号

n_clusters_205 = len(np.unique(labels_205))  # 第0类样本有几个簇

need_sampled_cluster_205 = []  # 需要下采样的簇号

cluster_to_num_205 = Counter(labels_205)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_205.keys():
    if cluster_to_num_205[k] > len(labels_205) // n_clusters_205:
        need_sampled_cluster_205.append(k)

need_sampled_losses_205 = [[] for _ in range(len(need_sampled_cluster_205))]

for i in range(len(need_sampled_cluster_205)):
    for j in range(len(labels_205)):
        if labels_205[j] == need_sampled_cluster_205[i]:
            need_sampled_losses_205[i].append(label_205_loss[j][0])

for j in range(len(labels_205)):
    if labels_205[j] not in need_sampled_cluster_205:
        selected_sampled_losses_205.append(label_205_loss[j][0])

for loss in need_sampled_losses_205:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_205.append(lo)
print(len(label_205_loss))
print(len(set(selected_sampled_losses_205)))

for loss in list(set(selected_sampled_losses_205)):
    for img in label_205_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第206类的采样=================================================
selected_sampled_losses_206 = []
ms_206 = MeanShift()
ms_206.fit(label_206_loss)

labels_206 = ms_206.labels_  # 第0类中的每个样本对应的簇号

n_clusters_206 = len(np.unique(labels_206))  # 第0类样本有几个簇

need_sampled_cluster_206 = []  # 需要下采样的簇号

cluster_to_num_206 = Counter(labels_206)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_206.keys():
    if cluster_to_num_206[k] > len(labels_206) // n_clusters_206:
        need_sampled_cluster_206.append(k)

need_sampled_losses_206 = [[] for _ in range(len(need_sampled_cluster_206))]

for i in range(len(need_sampled_cluster_206)):
    for j in range(len(labels_206)):
        if labels_206[j] == need_sampled_cluster_206[i]:
            need_sampled_losses_206[i].append(label_206_loss[j][0])

for j in range(len(labels_206)):
    if labels_206[j] not in need_sampled_cluster_206:
        selected_sampled_losses_206.append(label_206_loss[j][0])

for loss in need_sampled_losses_206:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_206.append(lo)
print(len(label_206_loss))
print(len(set(selected_sampled_losses_206)))

for loss in list(set(selected_sampled_losses_206)):
    for img in label_206_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第207类的采样=================================================
selected_sampled_losses_207 = []
ms_207 = MeanShift()
ms_207.fit(label_207_loss)

labels_207 = ms_207.labels_  # 第0类中的每个样本对应的簇号

n_clusters_207 = len(np.unique(labels_207))  # 第0类样本有几个簇

need_sampled_cluster_207 = []  # 需要下采样的簇号

cluster_to_num_207 = Counter(labels_207)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_207.keys():
    if cluster_to_num_207[k] > len(labels_207) // n_clusters_207:
        need_sampled_cluster_207.append(k)

need_sampled_losses_207 = [[] for _ in range(len(need_sampled_cluster_207))]

for i in range(len(need_sampled_cluster_207)):
    for j in range(len(labels_207)):
        if labels_207[j] == need_sampled_cluster_207[i]:
            need_sampled_losses_207[i].append(label_207_loss[j][0])

for j in range(len(labels_207)):
    if labels_207[j] not in need_sampled_cluster_207:
        selected_sampled_losses_207.append(label_207_loss[j][0])

for loss in need_sampled_losses_207:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_207.append(lo)
print(len(label_207_loss))
print(len(set(selected_sampled_losses_207)))

for loss in list(set(selected_sampled_losses_207)):
    for img in label_207_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第208类的采样=================================================
selected_sampled_losses_208 = []
ms_208 = MeanShift()
ms_208.fit(label_208_loss)

labels_208 = ms_208.labels_  # 第0类中的每个样本对应的簇号

n_clusters_208 = len(np.unique(labels_208))  # 第0类样本有几个簇

need_sampled_cluster_208 = []  # 需要下采样的簇号

cluster_to_num_208 = Counter(labels_208)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_208.keys():
    if cluster_to_num_208[k] > len(labels_208) // n_clusters_208:
        need_sampled_cluster_208.append(k)

need_sampled_losses_208 = [[] for _ in range(len(need_sampled_cluster_208))]

for i in range(len(need_sampled_cluster_208)):
    for j in range(len(labels_208)):
        if labels_208[j] == need_sampled_cluster_208[i]:
            need_sampled_losses_208[i].append(label_208_loss[j][0])

for j in range(len(labels_208)):
    if labels_208[j] not in need_sampled_cluster_208:
        selected_sampled_losses_208.append(label_208_loss[j][0])

for loss in need_sampled_losses_208:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_208.append(lo)
print(len(label_208_loss))
print(len(set(selected_sampled_losses_208)))

for loss in list(set(selected_sampled_losses_208)):
    for img in label_208_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第209类的采样=================================================
selected_sampled_losses_209 = []
ms_209 = MeanShift()
ms_209.fit(label_209_loss)

labels_209 = ms_209.labels_  # 第0类中的每个样本对应的簇号

n_clusters_209 = len(np.unique(labels_209))  # 第0类样本有几个簇

need_sampled_cluster_209 = []  # 需要下采样的簇号

cluster_to_num_209 = Counter(labels_209)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_209.keys():
    if cluster_to_num_209[k] > len(labels_209) // n_clusters_209:
        need_sampled_cluster_209.append(k)

need_sampled_losses_209 = [[] for _ in range(len(need_sampled_cluster_209))]

for i in range(len(need_sampled_cluster_209)):
    for j in range(len(labels_209)):
        if labels_209[j] == need_sampled_cluster_209[i]:
            need_sampled_losses_209[i].append(label_209_loss[j][0])

for j in range(len(labels_209)):
    if labels_209[j] not in need_sampled_cluster_209:
        selected_sampled_losses_209.append(label_209_loss[j][0])

for loss in need_sampled_losses_209:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_209.append(lo)
print(len(label_209_loss))
print(len(set(selected_sampled_losses_209)))

for loss in list(set(selected_sampled_losses_209)):
    for img in label_209_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第210类的采样=================================================
selected_sampled_losses_210 = []
ms_210 = MeanShift()
ms_210.fit(label_210_loss)

labels_210 = ms_210.labels_  # 第0类中的每个样本对应的簇号

n_clusters_210 = len(np.unique(labels_210))  # 第0类样本有几个簇

need_sampled_cluster_210 = []  # 需要下采样的簇号

cluster_to_num_210 = Counter(labels_210)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_210.keys():
    if cluster_to_num_210[k] > len(labels_210) // n_clusters_210:
        need_sampled_cluster_210.append(k)

need_sampled_losses_210 = [[] for _ in range(len(need_sampled_cluster_210))]

for i in range(len(need_sampled_cluster_210)):
    for j in range(len(labels_210)):
        if labels_210[j] == need_sampled_cluster_210[i]:
            need_sampled_losses_210[i].append(label_210_loss[j][0])

for j in range(len(labels_210)):
    if labels_210[j] not in need_sampled_cluster_210:
        selected_sampled_losses_210.append(label_210_loss[j][0])

for loss in need_sampled_losses_210:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_210.append(lo)
print(len(label_210_loss))
print(len(set(selected_sampled_losses_210)))

for loss in list(set(selected_sampled_losses_210)):
    for img in label_210_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第211类的采样=================================================
selected_sampled_losses_211 = []

ms_211 = MeanShift()
ms_211.fit(label_211_loss)

labels_211 = ms_211.labels_  # 第0类中的每个样本对应的簇号

n_clusters_211 = len(np.unique(labels_211))  # 第0类样本有几个簇

need_sampled_cluster_211 = []  # 需要下采样的簇号

cluster_to_num_211 = Counter(labels_211)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_211.keys():
    if cluster_to_num_211[k] > len(labels_211) // n_clusters_211:
        need_sampled_cluster_211.append(k)

need_sampled_losses_211 = [[] for _ in range(len(need_sampled_cluster_211))]

for i in range(len(need_sampled_cluster_211)):
    for j in range(len(labels_211)):
        if labels_211[j] == need_sampled_cluster_211[i]:
            need_sampled_losses_211[i].append(label_211_loss[j][0])

for j in range(len(labels_211)):
    if labels_211[j] not in need_sampled_cluster_211:
        selected_sampled_losses_211.append(label_211_loss[j][0])

for loss in need_sampled_losses_211:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_211.append(lo)
print(len(label_211_loss))
print(len(set(selected_sampled_losses_211)))

for loss in list(set(selected_sampled_losses_211)):
    for img in label_211_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第212类的采样=================================================
selected_sampled_losses_212 = []
ms_212 = MeanShift()
ms_212.fit(label_212_loss)

labels_212 = ms_212.labels_  # 第0类中的每个样本对应的簇号

n_clusters_212 = len(np.unique(labels_212))  # 第0类样本有几个簇

need_sampled_cluster_212 = []  # 需要下采样的簇号

cluster_to_num_212 = Counter(labels_212)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_212.keys():
    if cluster_to_num_212[k] > len(labels_212) // n_clusters_212:
        need_sampled_cluster_212.append(k)

need_sampled_losses_212 = [[] for _ in range(len(need_sampled_cluster_212))]

for i in range(len(need_sampled_cluster_212)):
    for j in range(len(labels_212)):
        if labels_212[j] == need_sampled_cluster_212[i]:
            need_sampled_losses_212[i].append(label_212_loss[j][0])

for j in range(len(labels_212)):
    if labels_212[j] not in need_sampled_cluster_212:
        selected_sampled_losses_212.append(label_212_loss[j][0])

for loss in need_sampled_losses_212:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_212.append(lo)
print(len(label_212_loss))
print(len(set(selected_sampled_losses_212)))

for loss in list(set(selected_sampled_losses_212)):
    for img in label_212_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第213类的采样=================================================
selected_sampled_losses_213 = []
ms_213 = MeanShift()
ms_213.fit(label_213_loss)

labels_213 = ms_213.labels_  # 第0类中的每个样本对应的簇号

n_clusters_213 = len(np.unique(labels_213))  # 第0类样本有几个簇

need_sampled_cluster_213 = []  # 需要下采样的簇号

cluster_to_num_213 = Counter(labels_213)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_213.keys():
    if cluster_to_num_213[k] > len(labels_213) // n_clusters_213:
        need_sampled_cluster_213.append(k)

need_sampled_losses_213 = [[] for _ in range(len(need_sampled_cluster_213))]

for i in range(len(need_sampled_cluster_213)):
    for j in range(len(labels_213)):
        if labels_213[j] == need_sampled_cluster_213[i]:
            need_sampled_losses_213[i].append(label_213_loss[j][0])

for j in range(len(labels_213)):
    if labels_213[j] not in need_sampled_cluster_213:
        selected_sampled_losses_213.append(label_213_loss[j][0])

for loss in need_sampled_losses_213:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_213.append(lo)
print(len(label_213_loss))
print(len(set(selected_sampled_losses_213)))

for loss in list(set(selected_sampled_losses_213)):
    for img in label_213_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第214类的采样=================================================
selected_sampled_losses_214 = []
ms_214 = MeanShift()
ms_214.fit(label_214_loss)

labels_214 = ms_214.labels_  # 第0类中的每个样本对应的簇号

n_clusters_214 = len(np.unique(labels_214))  # 第0类样本有几个簇

need_sampled_cluster_214 = []  # 需要下采样的簇号

cluster_to_num_214 = Counter(labels_214)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_214.keys():
    if cluster_to_num_214[k] > len(labels_214) // n_clusters_214:
        need_sampled_cluster_214.append(k)

need_sampled_losses_214 = [[] for _ in range(len(need_sampled_cluster_214))]

for i in range(len(need_sampled_cluster_214)):
    for j in range(len(labels_214)):
        if labels_214[j] == need_sampled_cluster_214[i]:
            need_sampled_losses_214[i].append(label_214_loss[j][0])

for j in range(len(labels_214)):
    if labels_214[j] not in need_sampled_cluster_214:
        selected_sampled_losses_214.append(label_214_loss[j][0])

for loss in need_sampled_losses_214:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_214.append(lo)
print(len(label_214_loss))
print(len(set(selected_sampled_losses_214)))

for loss in list(set(selected_sampled_losses_214)):
    for img in label_214_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第215类的采样=================================================
selected_sampled_losses_215 = []
ms_215 = MeanShift()
ms_215.fit(label_215_loss)

labels_215 = ms_215.labels_  # 第0类中的每个样本对应的簇号

n_clusters_215 = len(np.unique(labels_215))  # 第0类样本有几个簇

need_sampled_cluster_215 = []  # 需要下采样的簇号

cluster_to_num_215 = Counter(labels_215)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_215.keys():
    if cluster_to_num_215[k] > len(labels_215) // n_clusters_215:
        need_sampled_cluster_215.append(k)

need_sampled_losses_215 = [[] for _ in range(len(need_sampled_cluster_215))]

for i in range(len(need_sampled_cluster_215)):
    for j in range(len(labels_215)):
        if labels_215[j] == need_sampled_cluster_215[i]:
            need_sampled_losses_215[i].append(label_215_loss[j][0])

for j in range(len(labels_215)):
    if labels_215[j] not in need_sampled_cluster_215:
        selected_sampled_losses_215.append(label_215_loss[j][0])

for loss in need_sampled_losses_215:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_215.append(lo)
print(len(label_215_loss))
print(len(set(selected_sampled_losses_215)))

for loss in list(set(selected_sampled_losses_215)):
    for img in label_215_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第216类的采样=================================================
selected_sampled_losses_216 = []
ms_216 = MeanShift()
ms_216.fit(label_216_loss)

labels_216 = ms_216.labels_  # 第0类中的每个样本对应的簇号

n_clusters_216 = len(np.unique(labels_216))  # 第0类样本有几个簇

need_sampled_cluster_216 = []  # 需要下采样的簇号

cluster_to_num_216 = Counter(labels_216)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_216.keys():
    if cluster_to_num_216[k] > len(labels_216) // n_clusters_216:
        need_sampled_cluster_216.append(k)

need_sampled_losses_216 = [[] for _ in range(len(need_sampled_cluster_216))]

for i in range(len(need_sampled_cluster_216)):
    for j in range(len(labels_216)):
        if labels_216[j] == need_sampled_cluster_216[i]:
            need_sampled_losses_216[i].append(label_216_loss[j][0])

for j in range(len(labels_216)):
    if labels_216[j] not in need_sampled_cluster_216:
        selected_sampled_losses_216.append(label_216_loss[j][0])

for loss in need_sampled_losses_216:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_216.append(lo)
print(len(label_216_loss))
print(len(set(selected_sampled_losses_216)))

for loss in list(set(selected_sampled_losses_216)):
    for img in label_216_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第217类的采样=================================================
selected_sampled_losses_217 = []
ms_217 = MeanShift()
ms_217.fit(label_217_loss)

labels_217 = ms_217.labels_  # 第0类中的每个样本对应的簇号

n_clusters_217 = len(np.unique(labels_217))  # 第0类样本有几个簇

need_sampled_cluster_217 = []  # 需要下采样的簇号

cluster_to_num_217 = Counter(labels_217)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_217.keys():
    if cluster_to_num_217[k] > len(labels_217) // n_clusters_217:
        need_sampled_cluster_217.append(k)

need_sampled_losses_217 = [[] for _ in range(len(need_sampled_cluster_217))]

for i in range(len(need_sampled_cluster_217)):
    for j in range(len(labels_217)):
        if labels_217[j] == need_sampled_cluster_217[i]:
            need_sampled_losses_217[i].append(label_217_loss[j][0])

for j in range(len(labels_217)):
    if labels_217[j] not in need_sampled_cluster_217:
        selected_sampled_losses_217.append(label_217_loss[j][0])

for loss in need_sampled_losses_217:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_217.append(lo)
print(len(label_217_loss))
print(len(set(selected_sampled_losses_217)))

for loss in list(set(selected_sampled_losses_217)):
    for img in label_217_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第218类的采样=================================================
selected_sampled_losses_218 = []
ms_218 = MeanShift()
ms_218.fit(label_218_loss)

labels_218 = ms_218.labels_  # 第0类中的每个样本对应的簇号

n_clusters_218 = len(np.unique(labels_218))  # 第0类样本有几个簇

need_sampled_cluster_218 = []  # 需要下采样的簇号

cluster_to_num_218 = Counter(labels_218)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_218.keys():
    if cluster_to_num_218[k] > len(labels_218) // n_clusters_218:
        need_sampled_cluster_218.append(k)

need_sampled_losses_218 = [[] for _ in range(len(need_sampled_cluster_218))]

for i in range(len(need_sampled_cluster_218)):
    for j in range(len(labels_218)):
        if labels_218[j] == need_sampled_cluster_218[i]:
            need_sampled_losses_218[i].append(label_218_loss[j][0])

for j in range(len(labels_218)):
    if labels_218[j] not in need_sampled_cluster_218:
        selected_sampled_losses_218.append(label_218_loss[j][0])

for loss in need_sampled_losses_218:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_218.append(lo)
print(len(label_218_loss))
print(len(set(selected_sampled_losses_218)))

for loss in list(set(selected_sampled_losses_218)):
    for img in label_218_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第219类的采样=================================================
selected_sampled_losses_219 = []
ms_219 = MeanShift()
ms_219.fit(label_219_loss)

labels_219 = ms_219.labels_  # 第0类中的每个样本对应的簇号

n_clusters_219 = len(np.unique(labels_219))  # 第0类样本有几个簇

need_sampled_cluster_219 = []  # 需要下采样的簇号

cluster_to_num_219 = Counter(labels_219)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_219.keys():
    if cluster_to_num_219[k] > len(labels_219) // n_clusters_219:
        need_sampled_cluster_219.append(k)

need_sampled_losses_219 = [[] for _ in range(len(need_sampled_cluster_219))]

for i in range(len(need_sampled_cluster_219)):
    for j in range(len(labels_219)):
        if labels_219[j] == need_sampled_cluster_219[i]:
            need_sampled_losses_219[i].append(label_219_loss[j][0])

for j in range(len(labels_219)):
    if labels_219[j] not in need_sampled_cluster_219:
        selected_sampled_losses_219.append(label_219_loss[j][0])

for loss in need_sampled_losses_219:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_219.append(lo)
print(len(label_219_loss))
print(len(set(selected_sampled_losses_219)))
for loss in list(set(selected_sampled_losses_219)):
    for img in label_219_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))

# ==================================================针对第220类的采样=================================================

selected_sampled_losses_220 = []

ms_220 = MeanShift()
ms_220.fit(label_220_loss)

labels_220 = ms_220.labels_  # 第_220类中的每个样本对应的簇号

n_clusters_220 = len(np.unique(labels_220))  # 第_220类样本有几个簇

need_sampled_cluster_220 = []  # 需要下采样的簇号

cluster_to_num_220 = Counter(labels_220)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_220.keys():
    if cluster_to_num_220[k] > len(labels_220) // n_clusters_220:
        need_sampled_cluster_220.append(k)

need_sampled_losses_220 = [[] for _ in range(len(need_sampled_cluster_220))]

for i in range(len(need_sampled_cluster_220)):
    for j in range(len(labels_220)):
        if labels_220[j] == need_sampled_cluster_220[i]:
            need_sampled_losses_220[i].append(label_220_loss[j][0])

for j in range(len(labels_220)):
    if labels_220[j] not in need_sampled_cluster_220:
        selected_sampled_losses_220.append(label_220_loss[j][0])

for loss in need_sampled_losses_220:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_220.append(lo)
print(len(label_220_loss))
print(len(set(selected_sampled_losses_220)))

for loss in list(set(selected_sampled_losses_220)):
    for img in label_220_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第221类的采样=================================================
selected_sampled_losses_221 = []

ms_221 = MeanShift()
ms_221.fit(label_221_loss)

labels_221 = ms_221.labels_  # 第0类中的每个样本对应的簇号

n_clusters_221 = len(np.unique(labels_221))  # 第0类样本有几个簇

need_sampled_cluster_221 = []  # 需要下采样的簇号

cluster_to_num_221 = Counter(labels_221)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_221.keys():
    if cluster_to_num_221[k] > len(labels_221) // n_clusters_221:
        need_sampled_cluster_221.append(k)

need_sampled_losses_221 = [[] for _ in range(len(need_sampled_cluster_221))]

for i in range(len(need_sampled_cluster_221)):
    for j in range(len(labels_221)):
        if labels_221[j] == need_sampled_cluster_221[i]:
            need_sampled_losses_221[i].append(label_221_loss[j][0])

for j in range(len(labels_221)):
    if labels_221[j] not in need_sampled_cluster_221:
        selected_sampled_losses_221.append(label_221_loss[j][0])

for loss in need_sampled_losses_221:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_221.append(lo)
print(len(label_221_loss))
print(len(set(selected_sampled_losses_221)))

for loss in list(set(selected_sampled_losses_221)):
    for img in label_221_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第222类的采样=================================================
selected_sampled_losses_222 = []

ms_222 = MeanShift()
ms_222.fit(label_222_loss)

labels_222 = ms_222.labels_  # 第0类中的每个样本对应的簇号

n_clusters_222 = len(np.unique(labels_222))  # 第0类样本有几个簇

need_sampled_cluster_222 = []  # 需要下采样的簇号

cluster_to_num_222 = Counter(labels_222)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_222.keys():
    if cluster_to_num_222[k] > len(labels_222) // n_clusters_222:
        need_sampled_cluster_222.append(k)

need_sampled_losses_222 = [[] for _ in range(len(need_sampled_cluster_222))]

for i in range(len(need_sampled_cluster_222)):
    for j in range(len(labels_222)):
        if labels_222[j] == need_sampled_cluster_222[i]:
            need_sampled_losses_222[i].append(label_222_loss[j][0])

for j in range(len(labels_222)):
    if labels_222[j] not in need_sampled_cluster_222:
        selected_sampled_losses_222.append(label_222_loss[j][0])

for loss in need_sampled_losses_222:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_222.append(lo)
print(len(label_222_loss))
print(len(set(selected_sampled_losses_222)))

for loss in list(set(selected_sampled_losses_222)):
    for img in label_222_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第223类的采样=================================================
selected_sampled_losses_223 = []

ms_223 = MeanShift()
ms_223.fit(label_223_loss)

labels_223 = ms_223.labels_  # 第0类中的每个样本对应的簇号

n_clusters_223 = len(np.unique(labels_223))  # 第0类样本有几个簇

need_sampled_cluster_223 = []  # 需要下采样的簇号

cluster_to_num_223 = Counter(labels_223)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_223.keys():
    if cluster_to_num_223[k] > len(labels_223) // n_clusters_223:
        need_sampled_cluster_223.append(k)

need_sampled_losses_223 = [[] for _ in range(len(need_sampled_cluster_223))]

for i in range(len(need_sampled_cluster_223)):
    for j in range(len(labels_223)):
        if labels_223[j] == need_sampled_cluster_223[i]:
            need_sampled_losses_223[i].append(label_223_loss[j][0])

for j in range(len(labels_223)):
    if labels_223[j] not in need_sampled_cluster_223:
        selected_sampled_losses_223.append(label_223_loss[j][0])

for loss in need_sampled_losses_223:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_223.append(lo)
print(len(label_223_loss))
print(len(set(selected_sampled_losses_223)))

for loss in list(set(selected_sampled_losses_223)):
    for img in label_223_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第224类的采样=================================================
selected_sampled_losses_224 = []
ms_224 = MeanShift()
ms_224.fit(label_224_loss)

labels_224 = ms_224.labels_  # 第0类中的每个样本对应的簇号

n_clusters_224 = len(np.unique(labels_224))  # 第0类样本有几个簇

need_sampled_cluster_224 = []  # 需要下采样的簇号

cluster_to_num_224 = Counter(labels_224)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_224.keys():
    if cluster_to_num_224[k] > len(labels_224) // n_clusters_224:
        need_sampled_cluster_224.append(k)

need_sampled_losses_224 = [[] for _ in range(len(need_sampled_cluster_224))]

for i in range(len(need_sampled_cluster_224)):
    for j in range(len(labels_224)):
        if labels_224[j] == need_sampled_cluster_224[i]:
            need_sampled_losses_224[i].append(label_224_loss[j][0])

for j in range(len(labels_224)):
    if labels_224[j] not in need_sampled_cluster_224:
        selected_sampled_losses_224.append(label_224_loss[j][0])

for loss in need_sampled_losses_224:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_224.append(lo)
print(len(label_224_loss))
print(len(set(selected_sampled_losses_224)))

for loss in list(set(selected_sampled_losses_224)):
    for img in label_224_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第225类的采样=================================================
selected_sampled_losses_225 = []
ms_225 = MeanShift()
ms_225.fit(label_225_loss)

labels_225 = ms_225.labels_  # 第0类中的每个样本对应的簇号

n_clusters_225 = len(np.unique(labels_225))  # 第0类样本有几个簇

need_sampled_cluster_225 = []  # 需要下采样的簇号

cluster_to_num_225 = Counter(labels_225)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_225.keys():
    if cluster_to_num_225[k] > len(labels_225) // n_clusters_225:
        need_sampled_cluster_225.append(k)

need_sampled_losses_225 = [[] for _ in range(len(need_sampled_cluster_225))]

for i in range(len(need_sampled_cluster_225)):
    for j in range(len(labels_225)):
        if labels_225[j] == need_sampled_cluster_225[i]:
            need_sampled_losses_225[i].append(label_225_loss[j][0])

for j in range(len(labels_225)):
    if labels_225[j] not in need_sampled_cluster_225:
        selected_sampled_losses_225.append(label_225_loss[j][0])

for loss in need_sampled_losses_225:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_225.append(lo)
print(len(label_225_loss))
print(len(set(selected_sampled_losses_225)))

for loss in list(set(selected_sampled_losses_225)):
    for img in label_225_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第226类的采样=================================================
selected_sampled_losses_226 = []
ms_226 = MeanShift()
ms_226.fit(label_226_loss)

labels_226 = ms_226.labels_  # 第0类中的每个样本对应的簇号

n_clusters_226 = len(np.unique(labels_226))  # 第0类样本有几个簇

need_sampled_cluster_226 = []  # 需要下采样的簇号

cluster_to_num_226 = Counter(labels_226)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_226.keys():
    if cluster_to_num_226[k] > len(labels_226) // n_clusters_226:
        need_sampled_cluster_226.append(k)

need_sampled_losses_226 = [[] for _ in range(len(need_sampled_cluster_226))]

for i in range(len(need_sampled_cluster_226)):
    for j in range(len(labels_226)):
        if labels_226[j] == need_sampled_cluster_226[i]:
            need_sampled_losses_226[i].append(label_226_loss[j][0])

for j in range(len(labels_226)):
    if labels_226[j] not in need_sampled_cluster_226:
        selected_sampled_losses_226.append(label_226_loss[j][0])

for loss in need_sampled_losses_226:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_226.append(lo)
print(len(label_226_loss))
print(len(set(selected_sampled_losses_226)))

for loss in list(set(selected_sampled_losses_226)):
    for img in label_226_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第227类的采样=================================================
selected_sampled_losses_227 = []
ms_227 = MeanShift()
ms_227.fit(label_227_loss)

labels_227 = ms_227.labels_  # 第0类中的每个样本对应的簇号

n_clusters_227 = len(np.unique(labels_227))  # 第0类样本有几个簇

need_sampled_cluster_227 = []  # 需要下采样的簇号

cluster_to_num_227 = Counter(labels_227)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_227.keys():
    if cluster_to_num_227[k] > len(labels_227) // n_clusters_227:
        need_sampled_cluster_227.append(k)

need_sampled_losses_227 = [[] for _ in range(len(need_sampled_cluster_227))]

for i in range(len(need_sampled_cluster_227)):
    for j in range(len(labels_227)):
        if labels_227[j] == need_sampled_cluster_227[i]:
            need_sampled_losses_227[i].append(label_227_loss[j][0])

for j in range(len(labels_227)):
    if labels_227[j] not in need_sampled_cluster_227:
        selected_sampled_losses_227.append(label_227_loss[j][0])

for loss in need_sampled_losses_227:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_227.append(lo)
print(len(label_227_loss))
print(len(set(selected_sampled_losses_227)))

for loss in list(set(selected_sampled_losses_227)):
    for img in label_227_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第228类的采样=================================================
selected_sampled_losses_228 = []
ms_228 = MeanShift()
ms_228.fit(label_228_loss)

labels_228 = ms_228.labels_  # 第0类中的每个样本对应的簇号

n_clusters_228 = len(np.unique(labels_228))  # 第0类样本有几个簇

need_sampled_cluster_228 = []  # 需要下采样的簇号

cluster_to_num_228 = Counter(labels_228)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_228.keys():
    if cluster_to_num_228[k] > len(labels_228) // n_clusters_228:
        need_sampled_cluster_228.append(k)

need_sampled_losses_228 = [[] for _ in range(len(need_sampled_cluster_228))]

for i in range(len(need_sampled_cluster_228)):
    for j in range(len(labels_228)):
        if labels_228[j] == need_sampled_cluster_228[i]:
            need_sampled_losses_228[i].append(label_228_loss[j][0])

for j in range(len(labels_228)):
    if labels_228[j] not in need_sampled_cluster_228:
        selected_sampled_losses_228.append(label_228_loss[j][0])

for loss in need_sampled_losses_228:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_228.append(lo)
print(len(label_228_loss))
print(len(set(selected_sampled_losses_228)))

for loss in list(set(selected_sampled_losses_228)):
    for img in label_228_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第229类的采样=================================================
selected_sampled_losses_229 = []
ms_229 = MeanShift()
ms_229.fit(label_229_loss)

labels_229 = ms_229.labels_  # 第0类中的每个样本对应的簇号

n_clusters_229 = len(np.unique(labels_229))  # 第0类样本有几个簇

need_sampled_cluster_229 = []  # 需要下采样的簇号

cluster_to_num_229 = Counter(labels_229)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_229.keys():
    if cluster_to_num_229[k] > len(labels_229) // n_clusters_229:
        need_sampled_cluster_229.append(k)

need_sampled_losses_229 = [[] for _ in range(len(need_sampled_cluster_229))]

for i in range(len(need_sampled_cluster_229)):
    for j in range(len(labels_229)):
        if labels_229[j] == need_sampled_cluster_229[i]:
            need_sampled_losses_229[i].append(label_229_loss[j][0])

for j in range(len(labels_229)):
    if labels_229[j] not in need_sampled_cluster_229:
        selected_sampled_losses_229.append(label_229_loss[j][0])

for loss in need_sampled_losses_229:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_229.append(lo)
print(len(label_229_loss))
print(len(set(selected_sampled_losses_229)))

for loss in list(set(selected_sampled_losses_229)):
    for img in label_229_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第230类的采样=================================================
selected_sampled_losses_230 = []
ms_230 = MeanShift()
ms_230.fit(label_230_loss)

labels_230 = ms_230.labels_  # 第0类中的每个样本对应的簇号

n_clusters_230 = len(np.unique(labels_230))  # 第0类样本有几个簇

need_sampled_cluster_230 = []  # 需要下采样的簇号

cluster_to_num_230 = Counter(labels_230)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_230.keys():
    if cluster_to_num_230[k] > len(labels_230) // n_clusters_230:
        need_sampled_cluster_230.append(k)

need_sampled_losses_230 = [[] for _ in range(len(need_sampled_cluster_230))]

for i in range(len(need_sampled_cluster_230)):
    for j in range(len(labels_230)):
        if labels_230[j] == need_sampled_cluster_230[i]:
            need_sampled_losses_230[i].append(label_230_loss[j][0])

for j in range(len(labels_230)):
    if labels_230[j] not in need_sampled_cluster_230:
        selected_sampled_losses_230.append(label_230_loss[j][0])

for loss in need_sampled_losses_230:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_230.append(lo)
print(len(label_230_loss))
print(len(set(selected_sampled_losses_230)))

for loss in list(set(selected_sampled_losses_230)):
    for img in label_230_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第231类的采样=================================================
selected_sampled_losses_231 = []

ms_231 = MeanShift()
ms_231.fit(label_231_loss)

labels_231 = ms_231.labels_  # 第0类中的每个样本对应的簇号

n_clusters_231 = len(np.unique(labels_231))  # 第0类样本有几个簇

need_sampled_cluster_231 = []  # 需要下采样的簇号

cluster_to_num_231 = Counter(labels_231)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_231.keys():
    if cluster_to_num_231[k] > len(labels_231) // n_clusters_231:
        need_sampled_cluster_231.append(k)

need_sampled_losses_231 = [[] for _ in range(len(need_sampled_cluster_231))]

for i in range(len(need_sampled_cluster_231)):
    for j in range(len(labels_231)):
        if labels_231[j] == need_sampled_cluster_231[i]:
            need_sampled_losses_231[i].append(label_231_loss[j][0])

for j in range(len(labels_231)):
    if labels_231[j] not in need_sampled_cluster_231:
        selected_sampled_losses_231.append(label_231_loss[j][0])

for loss in need_sampled_losses_231:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_231.append(lo)
print(len(label_231_loss))
print(len(set(selected_sampled_losses_231)))

for loss in list(set(selected_sampled_losses_231)):
    for img in label_231_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第232类的采样=================================================
selected_sampled_losses_232 = []
ms_232 = MeanShift()
ms_232.fit(label_232_loss)

labels_232 = ms_232.labels_  # 第0类中的每个样本对应的簇号

n_clusters_232 = len(np.unique(labels_232))  # 第0类样本有几个簇

need_sampled_cluster_232 = []  # 需要下采样的簇号

cluster_to_num_232 = Counter(labels_232)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_232.keys():
    if cluster_to_num_232[k] > len(labels_232) // n_clusters_232:
        need_sampled_cluster_232.append(k)

need_sampled_losses_232 = [[] for _ in range(len(need_sampled_cluster_232))]

for i in range(len(need_sampled_cluster_232)):
    for j in range(len(labels_232)):
        if labels_232[j] == need_sampled_cluster_232[i]:
            need_sampled_losses_232[i].append(label_232_loss[j][0])

for j in range(len(labels_232)):
    if labels_232[j] not in need_sampled_cluster_232:
        selected_sampled_losses_232.append(label_232_loss[j][0])

for loss in need_sampled_losses_232:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_232.append(lo)
print(len(label_232_loss))
print(len(set(selected_sampled_losses_232)))

for loss in list(set(selected_sampled_losses_232)):
    for img in label_232_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第233类的采样=================================================
selected_sampled_losses_233 = []
ms_233 = MeanShift()
ms_233.fit(label_233_loss)

labels_233 = ms_233.labels_  # 第0类中的每个样本对应的簇号

n_clusters_233 = len(np.unique(labels_233))  # 第0类样本有几个簇

need_sampled_cluster_233 = []  # 需要下采样的簇号

cluster_to_num_233 = Counter(labels_233)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_233.keys():
    if cluster_to_num_233[k] > len(labels_233) // n_clusters_233:
        need_sampled_cluster_233.append(k)

need_sampled_losses_233 = [[] for _ in range(len(need_sampled_cluster_233))]

for i in range(len(need_sampled_cluster_233)):
    for j in range(len(labels_233)):
        if labels_233[j] == need_sampled_cluster_233[i]:
            need_sampled_losses_233[i].append(label_233_loss[j][0])

for j in range(len(labels_233)):
    if labels_233[j] not in need_sampled_cluster_233:
        selected_sampled_losses_233.append(label_233_loss[j][0])

for loss in need_sampled_losses_233:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_233.append(lo)
print(len(label_233_loss))
print(len(set(selected_sampled_losses_233)))

for loss in list(set(selected_sampled_losses_233)):
    for img in label_233_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第234类的采样=================================================
selected_sampled_losses_234 = []
ms_234 = MeanShift()
ms_234.fit(label_234_loss)

labels_234 = ms_234.labels_  # 第0类中的每个样本对应的簇号

n_clusters_234 = len(np.unique(labels_234))  # 第0类样本有几个簇

need_sampled_cluster_234 = []  # 需要下采样的簇号

cluster_to_num_234 = Counter(labels_234)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_234.keys():
    if cluster_to_num_234[k] > len(labels_234) // n_clusters_234:
        need_sampled_cluster_234.append(k)

need_sampled_losses_234 = [[] for _ in range(len(need_sampled_cluster_234))]

for i in range(len(need_sampled_cluster_234)):
    for j in range(len(labels_234)):
        if labels_234[j] == need_sampled_cluster_234[i]:
            need_sampled_losses_234[i].append(label_234_loss[j][0])

for j in range(len(labels_234)):
    if labels_234[j] not in need_sampled_cluster_234:
        selected_sampled_losses_234.append(label_234_loss[j][0])

for loss in need_sampled_losses_234:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_234.append(lo)
print(len(label_234_loss))
print(len(set(selected_sampled_losses_234)))

for loss in list(set(selected_sampled_losses_234)):
    for img in label_234_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第235类的采样=================================================
selected_sampled_losses_235 = []
ms_235 = MeanShift()
ms_235.fit(label_235_loss)

labels_235 = ms_235.labels_  # 第0类中的每个样本对应的簇号

n_clusters_235 = len(np.unique(labels_235))  # 第0类样本有几个簇

need_sampled_cluster_235 = []  # 需要下采样的簇号

cluster_to_num_235 = Counter(labels_235)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_235.keys():
    if cluster_to_num_235[k] > len(labels_235) // n_clusters_235:
        need_sampled_cluster_235.append(k)

need_sampled_losses_235 = [[] for _ in range(len(need_sampled_cluster_235))]

for i in range(len(need_sampled_cluster_235)):
    for j in range(len(labels_235)):
        if labels_235[j] == need_sampled_cluster_235[i]:
            need_sampled_losses_235[i].append(label_235_loss[j][0])

for j in range(len(labels_235)):
    if labels_235[j] not in need_sampled_cluster_235:
        selected_sampled_losses_235.append(label_235_loss[j][0])

for loss in need_sampled_losses_235:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_235.append(lo)
print(len(label_235_loss))
print(len(set(selected_sampled_losses_235)))

for loss in list(set(selected_sampled_losses_235)):
    for img in label_235_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第236类的采样=================================================
selected_sampled_losses_236 = []
ms_236 = MeanShift()
ms_236.fit(label_236_loss)

labels_236 = ms_236.labels_  # 第0类中的每个样本对应的簇号

n_clusters_236 = len(np.unique(labels_236))  # 第0类样本有几个簇

need_sampled_cluster_236 = []  # 需要下采样的簇号

cluster_to_num_236 = Counter(labels_236)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_236.keys():
    if cluster_to_num_236[k] > len(labels_236) // n_clusters_236:
        need_sampled_cluster_236.append(k)

need_sampled_losses_236 = [[] for _ in range(len(need_sampled_cluster_236))]

for i in range(len(need_sampled_cluster_236)):
    for j in range(len(labels_236)):
        if labels_236[j] == need_sampled_cluster_236[i]:
            need_sampled_losses_236[i].append(label_236_loss[j][0])

for j in range(len(labels_236)):
    if labels_236[j] not in need_sampled_cluster_236:
        selected_sampled_losses_236.append(label_236_loss[j][0])

for loss in need_sampled_losses_236:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_236.append(lo)
print(len(label_236_loss))
print(len(set(selected_sampled_losses_236)))

for loss in list(set(selected_sampled_losses_236)):
    for img in label_236_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第237类的采样=================================================
selected_sampled_losses_237 = []
ms_237 = MeanShift()
ms_237.fit(label_237_loss)

labels_237 = ms_237.labels_  # 第0类中的每个样本对应的簇号

n_clusters_237 = len(np.unique(labels_237))  # 第0类样本有几个簇

need_sampled_cluster_237 = []  # 需要下采样的簇号

cluster_to_num_237 = Counter(labels_237)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_237.keys():
    if cluster_to_num_237[k] > len(labels_237) // n_clusters_237:
        need_sampled_cluster_237.append(k)

need_sampled_losses_237 = [[] for _ in range(len(need_sampled_cluster_237))]

for i in range(len(need_sampled_cluster_237)):
    for j in range(len(labels_237)):
        if labels_237[j] == need_sampled_cluster_237[i]:
            need_sampled_losses_237[i].append(label_237_loss[j][0])

for j in range(len(labels_237)):
    if labels_237[j] not in need_sampled_cluster_237:
        selected_sampled_losses_237.append(label_237_loss[j][0])

for loss in need_sampled_losses_237:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_237.append(lo)
print(len(label_237_loss))
print(len(set(selected_sampled_losses_237)))

for loss in list(set(selected_sampled_losses_237)):
    for img in label_237_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第238类的采样=================================================
selected_sampled_losses_238 = []
ms_238 = MeanShift()
ms_238.fit(label_238_loss)

labels_238 = ms_238.labels_  # 第0类中的每个样本对应的簇号

n_clusters_238 = len(np.unique(labels_238))  # 第0类样本有几个簇

need_sampled_cluster_238 = []  # 需要下采样的簇号

cluster_to_num_238 = Counter(labels_238)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_238.keys():
    if cluster_to_num_238[k] > len(labels_238) // n_clusters_238:
        need_sampled_cluster_238.append(k)

need_sampled_losses_238 = [[] for _ in range(len(need_sampled_cluster_238))]

for i in range(len(need_sampled_cluster_238)):
    for j in range(len(labels_238)):
        if labels_238[j] == need_sampled_cluster_238[i]:
            need_sampled_losses_238[i].append(label_238_loss[j][0])

for j in range(len(labels_238)):
    if labels_238[j] not in need_sampled_cluster_238:
        selected_sampled_losses_238.append(label_238_loss[j][0])

for loss in need_sampled_losses_238:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_238.append(lo)
print(len(label_238_loss))
print(len(set(selected_sampled_losses_238)))

for loss in list(set(selected_sampled_losses_238)):
    for img in label_238_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第239类的采样=================================================
selected_sampled_losses_239 = []
ms_239 = MeanShift()
ms_239.fit(label_239_loss)

labels_239 = ms_239.labels_  # 第0类中的每个样本对应的簇号

n_clusters_239 = len(np.unique(labels_239))  # 第0类样本有几个簇

need_sampled_cluster_239 = []  # 需要下采样的簇号

cluster_to_num_239 = Counter(labels_239)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_239.keys():
    if cluster_to_num_239[k] > len(labels_239) // n_clusters_239:
        need_sampled_cluster_239.append(k)

need_sampled_losses_239 = [[] for _ in range(len(need_sampled_cluster_239))]

for i in range(len(need_sampled_cluster_239)):
    for j in range(len(labels_239)):
        if labels_239[j] == need_sampled_cluster_239[i]:
            need_sampled_losses_239[i].append(label_239_loss[j][0])

for j in range(len(labels_239)):
    if labels_239[j] not in need_sampled_cluster_239:
        selected_sampled_losses_239.append(label_239_loss[j][0])

for loss in need_sampled_losses_239:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_239.append(lo)
print(len(label_239_loss))
print(len(set(selected_sampled_losses_239)))
for loss in list(set(selected_sampled_losses_239)):
    for img in label_239_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))





# ==================================================针对第240类的采样=================================================
selected_sampled_losses_240 = []
ms_240 = MeanShift()
ms_240.fit(label_240_loss)

labels_240 = ms_240.labels_  # 第0类中的每个样本对应的簇号

n_clusters_240 = len(np.unique(labels_240))  # 第0类样本有几个簇

need_sampled_cluster_240 = []  # 需要下采样的簇号

cluster_to_num_240 = Counter(labels_240)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_240.keys():
    if cluster_to_num_240[k] > len(labels_240) // n_clusters_240:
        need_sampled_cluster_240.append(k)

need_sampled_losses_240 = [[] for _ in range(len(need_sampled_cluster_240))]

for i in range(len(need_sampled_cluster_240)):
    for j in range(len(labels_240)):
        if labels_240[j] == need_sampled_cluster_240[i]:
            need_sampled_losses_240[i].append(label_240_loss[j][0])

for j in range(len(labels_240)):
    if labels_240[j] not in need_sampled_cluster_240:
        selected_sampled_losses_240.append(label_240_loss[j][0])

for loss in need_sampled_losses_240:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_240.append(lo)
print(len(label_240_loss))
print(len(set(selected_sampled_losses_240)))

for loss in list(set(selected_sampled_losses_240)):
    for img in label_240_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第241类的采样=================================================
selected_sampled_losses_241 = []

ms_241 = MeanShift()
ms_241.fit(label_241_loss)

labels_241 = ms_241.labels_  # 第0类中的每个样本对应的簇号

n_clusters_241 = len(np.unique(labels_241))  # 第0类样本有几个簇

need_sampled_cluster_241 = []  # 需要下采样的簇号

cluster_to_num_241 = Counter(labels_241)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_241.keys():
    if cluster_to_num_241[k] > len(labels_241) // n_clusters_241:
        need_sampled_cluster_241.append(k)

need_sampled_losses_241 = [[] for _ in range(len(need_sampled_cluster_241))]

for i in range(len(need_sampled_cluster_241)):
    for j in range(len(labels_241)):
        if labels_241[j] == need_sampled_cluster_241[i]:
            need_sampled_losses_241[i].append(label_241_loss[j][0])

for j in range(len(labels_241)):
    if labels_241[j] not in need_sampled_cluster_241:
        selected_sampled_losses_241.append(label_241_loss[j][0])

for loss in need_sampled_losses_241:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_241.append(lo)
print(len(label_241_loss))
print(len(set(selected_sampled_losses_241)))

for loss in list(set(selected_sampled_losses_241)):
    for img in label_241_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第242类的采样=================================================
selected_sampled_losses_242 = []
ms_242 = MeanShift()
ms_242.fit(label_242_loss)

labels_242 = ms_242.labels_  # 第0类中的每个样本对应的簇号

n_clusters_242 = len(np.unique(labels_242))  # 第0类样本有几个簇

need_sampled_cluster_242 = []  # 需要下采样的簇号

cluster_to_num_242 = Counter(labels_242)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_242.keys():
    if cluster_to_num_242[k] > len(labels_242) // n_clusters_242:
        need_sampled_cluster_242.append(k)

need_sampled_losses_242 = [[] for _ in range(len(need_sampled_cluster_242))]

for i in range(len(need_sampled_cluster_242)):
    for j in range(len(labels_242)):
        if labels_242[j] == need_sampled_cluster_242[i]:
            need_sampled_losses_242[i].append(label_242_loss[j][0])

for j in range(len(labels_242)):
    if labels_242[j] not in need_sampled_cluster_242:
        selected_sampled_losses_242.append(label_242_loss[j][0])

for loss in need_sampled_losses_242:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_242.append(lo)
print(len(label_242_loss))
print(len(set(selected_sampled_losses_242)))

for loss in list(set(selected_sampled_losses_242)):
    for img in label_242_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第243类的采样=================================================
selected_sampled_losses_243 = []
ms_243 = MeanShift()
ms_243.fit(label_243_loss)

labels_243 = ms_243.labels_  # 第0类中的每个样本对应的簇号

n_clusters_243 = len(np.unique(labels_243))  # 第0类样本有几个簇

need_sampled_cluster_243 = []  # 需要下采样的簇号

cluster_to_num_243 = Counter(labels_243)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_243.keys():
    if cluster_to_num_243[k] > len(labels_243) // n_clusters_243:
        need_sampled_cluster_243.append(k)

need_sampled_losses_243 = [[] for _ in range(len(need_sampled_cluster_243))]

for i in range(len(need_sampled_cluster_243)):
    for j in range(len(labels_243)):
        if labels_243[j] == need_sampled_cluster_243[i]:
            need_sampled_losses_243[i].append(label_243_loss[j][0])

for j in range(len(labels_243)):
    if labels_243[j] not in need_sampled_cluster_243:
        selected_sampled_losses_243.append(label_243_loss[j][0])

for loss in need_sampled_losses_243:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_243.append(lo)
print(len(label_243_loss))
print(len(set(selected_sampled_losses_243)))

for loss in list(set(selected_sampled_losses_243)):
    for img in label_243_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第244类的采样=================================================
selected_sampled_losses_244 = []
ms_244 = MeanShift()
ms_244.fit(label_244_loss)

labels_244 = ms_244.labels_  # 第0类中的每个样本对应的簇号

n_clusters_244 = len(np.unique(labels_244))  # 第0类样本有几个簇

need_sampled_cluster_244 = []  # 需要下采样的簇号

cluster_to_num_244 = Counter(labels_244)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_244.keys():
    if cluster_to_num_244[k] > len(labels_244) // n_clusters_244:
        need_sampled_cluster_244.append(k)

need_sampled_losses_244 = [[] for _ in range(len(need_sampled_cluster_244))]

for i in range(len(need_sampled_cluster_244)):
    for j in range(len(labels_244)):
        if labels_244[j] == need_sampled_cluster_244[i]:
            need_sampled_losses_244[i].append(label_244_loss[j][0])

for j in range(len(labels_244)):
    if labels_244[j] not in need_sampled_cluster_244:
        selected_sampled_losses_244.append(label_244_loss[j][0])

for loss in need_sampled_losses_244:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_244.append(lo)
print(len(label_244_loss))
print(len(set(selected_sampled_losses_244)))

for loss in list(set(selected_sampled_losses_244)):
    for img in label_244_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第245类的采样=================================================
selected_sampled_losses_245 = []
ms_245 = MeanShift()
ms_245.fit(label_245_loss)

labels_245 = ms_245.labels_  # 第0类中的每个样本对应的簇号

n_clusters_245 = len(np.unique(labels_245))  # 第0类样本有几个簇

need_sampled_cluster_245 = []  # 需要下采样的簇号

cluster_to_num_245 = Counter(labels_245)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_245.keys():
    if cluster_to_num_245[k] > len(labels_245) // n_clusters_245:
        need_sampled_cluster_245.append(k)

need_sampled_losses_245 = [[] for _ in range(len(need_sampled_cluster_245))]

for i in range(len(need_sampled_cluster_245)):
    for j in range(len(labels_245)):
        if labels_245[j] == need_sampled_cluster_245[i]:
            need_sampled_losses_245[i].append(label_245_loss[j][0])

for j in range(len(labels_245)):
    if labels_245[j] not in need_sampled_cluster_245:
        selected_sampled_losses_245.append(label_245_loss[j][0])

for loss in need_sampled_losses_245:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_245.append(lo)
print(len(label_245_loss))
print(len(set(selected_sampled_losses_245)))

for loss in list(set(selected_sampled_losses_245)):
    for img in label_245_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第246类的采样=================================================
selected_sampled_losses_246 = []
ms_246 = MeanShift()
ms_246.fit(label_246_loss)

labels_246 = ms_246.labels_  # 第0类中的每个样本对应的簇号

n_clusters_246 = len(np.unique(labels_246))  # 第0类样本有几个簇

need_sampled_cluster_246 = []  # 需要下采样的簇号

cluster_to_num_246 = Counter(labels_246)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_246.keys():
    if cluster_to_num_246[k] > len(labels_246) // n_clusters_246:
        need_sampled_cluster_246.append(k)

need_sampled_losses_246 = [[] for _ in range(len(need_sampled_cluster_246))]

for i in range(len(need_sampled_cluster_246)):
    for j in range(len(labels_246)):
        if labels_246[j] == need_sampled_cluster_246[i]:
            need_sampled_losses_246[i].append(label_246_loss[j][0])

for j in range(len(labels_246)):
    if labels_246[j] not in need_sampled_cluster_246:
        selected_sampled_losses_246.append(label_246_loss[j][0])

for loss in need_sampled_losses_246:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_246.append(lo)
print(len(label_246_loss))
print(len(set(selected_sampled_losses_246)))

for loss in list(set(selected_sampled_losses_246)):
    for img in label_246_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第247类的采样=================================================
selected_sampled_losses_247 = []
ms_247 = MeanShift()
ms_247.fit(label_247_loss)

labels_247 = ms_247.labels_  # 第0类中的每个样本对应的簇号

n_clusters_247 = len(np.unique(labels_247))  # 第0类样本有几个簇

need_sampled_cluster_247 = []  # 需要下采样的簇号

cluster_to_num_247 = Counter(labels_247)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_247.keys():
    if cluster_to_num_247[k] > len(labels_247) // n_clusters_247:
        need_sampled_cluster_247.append(k)

need_sampled_losses_247 = [[] for _ in range(len(need_sampled_cluster_247))]

for i in range(len(need_sampled_cluster_247)):
    for j in range(len(labels_247)):
        if labels_247[j] == need_sampled_cluster_247[i]:
            need_sampled_losses_247[i].append(label_247_loss[j][0])

for j in range(len(labels_247)):
    if labels_247[j] not in need_sampled_cluster_247:
        selected_sampled_losses_247.append(label_247_loss[j][0])

for loss in need_sampled_losses_247:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_247.append(lo)
print(len(label_247_loss))
print(len(set(selected_sampled_losses_247)))

for loss in list(set(selected_sampled_losses_247)):
    for img in label_247_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第248类的采样=================================================
selected_sampled_losses_248 = []
ms_248 = MeanShift()
ms_248.fit(label_248_loss)

labels_248 = ms_248.labels_  # 第0类中的每个样本对应的簇号

n_clusters_248 = len(np.unique(labels_248))  # 第0类样本有几个簇

need_sampled_cluster_248 = []  # 需要下采样的簇号

cluster_to_num_248 = Counter(labels_248)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_248.keys():
    if cluster_to_num_248[k] > len(labels_248) // n_clusters_248:
        need_sampled_cluster_248.append(k)

need_sampled_losses_248 = [[] for _ in range(len(need_sampled_cluster_248))]

for i in range(len(need_sampled_cluster_248)):
    for j in range(len(labels_248)):
        if labels_248[j] == need_sampled_cluster_248[i]:
            need_sampled_losses_248[i].append(label_248_loss[j][0])

for j in range(len(labels_248)):
    if labels_248[j] not in need_sampled_cluster_248:
        selected_sampled_losses_248.append(label_248_loss[j][0])

for loss in need_sampled_losses_248:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_248.append(lo)
print(len(label_248_loss))
print(len(set(selected_sampled_losses_248)))

for loss in list(set(selected_sampled_losses_248)):
    for img in label_248_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第249类的采样=================================================
selected_sampled_losses_249 = []
ms_249 = MeanShift()
ms_249.fit(label_249_loss)

labels_249 = ms_249.labels_  # 第0类中的每个样本对应的簇号

n_clusters_249 = len(np.unique(labels_249))  # 第0类样本有几个簇

need_sampled_cluster_249 = []  # 需要下采样的簇号

cluster_to_num_249 = Counter(labels_249)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_249.keys():
    if cluster_to_num_249[k] > len(labels_249) // n_clusters_249:
        need_sampled_cluster_249.append(k)

need_sampled_losses_249 = [[] for _ in range(len(need_sampled_cluster_249))]

for i in range(len(need_sampled_cluster_249)):
    for j in range(len(labels_249)):
        if labels_249[j] == need_sampled_cluster_249[i]:
            need_sampled_losses_249[i].append(label_249_loss[j][0])

for j in range(len(labels_249)):
    if labels_249[j] not in need_sampled_cluster_249:
        selected_sampled_losses_249.append(label_249_loss[j][0])

for loss in need_sampled_losses_249:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_249.append(lo)
print(len(label_249_loss))
print(len(set(selected_sampled_losses_249)))
for loss in list(set(selected_sampled_losses_249)):
    for img in label_249_loss_to_img[loss]:
        imgs.append(img)
print(len(set(imgs)))



# ==================================================针对第250类的采样=================================================
selected_sampled_losses_250 = []
ms_250 = MeanShift()
ms_250.fit(label_250_loss)

labels_250 = ms_250.labels_  # 第0类中的每个样本对应的簇号

n_clusters_250 = len(np.unique(labels_250))  # 第0类样本有几个簇

need_sampled_cluster_250 = []  # 需要下采样的簇号

cluster_to_num_250 = Counter(labels_250)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_250.keys():
    if cluster_to_num_250[k] > len(labels_250) // n_clusters_250:
        need_sampled_cluster_250.append(k)

need_sampled_losses_250 = [[] for _ in range(len(need_sampled_cluster_250))]

for i in range(len(need_sampled_cluster_250)):
    for j in range(len(labels_250)):
        if labels_250[j] == need_sampled_cluster_250[i]:
            need_sampled_losses_250[i].append(label_250_loss[j][0])

for j in range(len(labels_250)):
    if labels_250[j] not in need_sampled_cluster_250:
        selected_sampled_losses_250.append(label_250_loss[j][0])

for loss in need_sampled_losses_250:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_250.append(lo)
print(len(label_250_loss))
print(len(set(selected_sampled_losses_250)))

for loss in list(set(selected_sampled_losses_250)):
    for img in label_250_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第251类的采样=================================================
selected_sampled_losses_251 = []

ms_251 = MeanShift()
ms_251.fit(label_251_loss)

labels_251 = ms_251.labels_  # 第0类中的每个样本对应的簇号

n_clusters_251 = len(np.unique(labels_251))  # 第0类样本有几个簇

need_sampled_cluster_251 = []  # 需要下采样的簇号

cluster_to_num_251 = Counter(labels_251)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_251.keys():
    if cluster_to_num_251[k] > len(labels_251) // n_clusters_251:
        need_sampled_cluster_251.append(k)

need_sampled_losses_251 = [[] for _ in range(len(need_sampled_cluster_251))]

for i in range(len(need_sampled_cluster_251)):
    for j in range(len(labels_251)):
        if labels_251[j] == need_sampled_cluster_251[i]:
            need_sampled_losses_251[i].append(label_251_loss[j][0])

for j in range(len(labels_251)):
    if labels_251[j] not in need_sampled_cluster_251:
        selected_sampled_losses_251.append(label_251_loss[j][0])

for loss in need_sampled_losses_251:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_251.append(lo)
print(len(label_251_loss))
print(len(set(selected_sampled_losses_251)))

for loss in list(set(selected_sampled_losses_251)):
    for img in label_251_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第252类的采样=================================================
selected_sampled_losses_252 = []
ms_252 = MeanShift()
ms_252.fit(label_252_loss)

labels_252 = ms_252.labels_  # 第0类中的每个样本对应的簇号

n_clusters_252 = len(np.unique(labels_252))  # 第0类样本有几个簇

need_sampled_cluster_252 = []  # 需要下采样的簇号

cluster_to_num_252 = Counter(labels_252)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_252.keys():
    if cluster_to_num_252[k] > len(labels_252) // n_clusters_252:
        need_sampled_cluster_252.append(k)

need_sampled_losses_252 = [[] for _ in range(len(need_sampled_cluster_252))]

for i in range(len(need_sampled_cluster_252)):
    for j in range(len(labels_252)):
        if labels_252[j] == need_sampled_cluster_252[i]:
            need_sampled_losses_252[i].append(label_252_loss[j][0])

for j in range(len(labels_252)):
    if labels_252[j] not in need_sampled_cluster_252:
        selected_sampled_losses_252.append(label_252_loss[j][0])

for loss in need_sampled_losses_252:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_252.append(lo)
print(len(label_252_loss))
print(len(set(selected_sampled_losses_252)))

for loss in list(set(selected_sampled_losses_252)):
    for img in label_252_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第253类的采样=================================================
selected_sampled_losses_253 = []
ms_253 = MeanShift()
ms_253.fit(label_253_loss)

labels_253 = ms_253.labels_  # 第0类中的每个样本对应的簇号

n_clusters_253 = len(np.unique(labels_253))  # 第0类样本有几个簇

need_sampled_cluster_253 = []  # 需要下采样的簇号

cluster_to_num_253 = Counter(labels_253)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_253.keys():
    if cluster_to_num_253[k] > len(labels_253) // n_clusters_253:
        need_sampled_cluster_253.append(k)

need_sampled_losses_253 = [[] for _ in range(len(need_sampled_cluster_253))]

for i in range(len(need_sampled_cluster_253)):
    for j in range(len(labels_253)):
        if labels_253[j] == need_sampled_cluster_253[i]:
            need_sampled_losses_253[i].append(label_253_loss[j][0])

for j in range(len(labels_253)):
    if labels_253[j] not in need_sampled_cluster_253:
        selected_sampled_losses_253.append(label_253_loss[j][0])

for loss in need_sampled_losses_253:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_253.append(lo)
print(len(label_253_loss))
print(len(set(selected_sampled_losses_253)))

for loss in list(set(selected_sampled_losses_253)):
    for img in label_253_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第254类的采样=================================================
selected_sampled_losses_254 = []
ms_254 = MeanShift()
ms_254.fit(label_254_loss)

labels_254 = ms_254.labels_  # 第0类中的每个样本对应的簇号

n_clusters_254 = len(np.unique(labels_254))  # 第0类样本有几个簇

need_sampled_cluster_254 = []  # 需要下采样的簇号

cluster_to_num_254 = Counter(labels_254)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_254.keys():
    if cluster_to_num_254[k] > len(labels_254) // n_clusters_254:
        need_sampled_cluster_254.append(k)

need_sampled_losses_254 = [[] for _ in range(len(need_sampled_cluster_254))]

for i in range(len(need_sampled_cluster_254)):
    for j in range(len(labels_254)):
        if labels_254[j] == need_sampled_cluster_254[i]:
            need_sampled_losses_254[i].append(label_254_loss[j][0])

for j in range(len(labels_254)):
    if labels_254[j] not in need_sampled_cluster_254:
        selected_sampled_losses_254.append(label_254_loss[j][0])

for loss in need_sampled_losses_254:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_254.append(lo)
print(len(label_254_loss))
print(len(set(selected_sampled_losses_254)))

for loss in list(set(selected_sampled_losses_254)):
    for img in label_254_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第255类的采样=================================================
selected_sampled_losses_255 = []
ms_255 = MeanShift()
ms_255.fit(label_255_loss)

labels_255 = ms_255.labels_  # 第0类中的每个样本对应的簇号

n_clusters_255 = len(np.unique(labels_255))  # 第0类样本有几个簇

need_sampled_cluster_255 = []  # 需要下采样的簇号

cluster_to_num_255 = Counter(labels_255)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_255.keys():
    if cluster_to_num_255[k] > len(labels_255) // n_clusters_255:
        need_sampled_cluster_255.append(k)

need_sampled_losses_255 = [[] for _ in range(len(need_sampled_cluster_255))]

for i in range(len(need_sampled_cluster_255)):
    for j in range(len(labels_255)):
        if labels_255[j] == need_sampled_cluster_255[i]:
            need_sampled_losses_255[i].append(label_255_loss[j][0])

for j in range(len(labels_255)):
    if labels_255[j] not in need_sampled_cluster_255:
        selected_sampled_losses_255.append(label_255_loss[j][0])

for loss in need_sampled_losses_255:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_255.append(lo)
print(len(label_255_loss))
print(len(set(selected_sampled_losses_255)))

for loss in list(set(selected_sampled_losses_255)):
    for img in label_255_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第256类的采样=================================================
selected_sampled_losses_256 = []
ms_256 = MeanShift()
ms_256.fit(label_256_loss)

labels_256 = ms_256.labels_  # 第0类中的每个样本对应的簇号

n_clusters_256 = len(np.unique(labels_256))  # 第0类样本有几个簇

need_sampled_cluster_256 = []  # 需要下采样的簇号

cluster_to_num_256 = Counter(labels_256)  # 每一个簇对应的个数，字典形式

for k in cluster_to_num_256.keys():
    if cluster_to_num_256[k] > len(labels_256) // n_clusters_256:
        need_sampled_cluster_256.append(k)

need_sampled_losses_256 = [[] for _ in range(len(need_sampled_cluster_256))]

for i in range(len(need_sampled_cluster_256)):
    for j in range(len(labels_256)):
        if labels_256[j] == need_sampled_cluster_256[i]:
            need_sampled_losses_256[i].append(label_256_loss[j][0])

for j in range(len(labels_256)):
    if labels_256[j] not in need_sampled_cluster_256:
        selected_sampled_losses_256.append(label_256_loss[j][0])

for loss in need_sampled_losses_256:
    loss = np.array(loss)
    # print(len(loss))
    sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
    # print(len(set(sampled_loss)))

    for lo in sampled_loss:
        selected_sampled_losses_256.append(lo)
print(len(label_256_loss))
print(len(set(selected_sampled_losses_256)))

for loss in list(set(selected_sampled_losses_256)):
    for img in label_256_loss_to_img[loss]:
        imgs.append(img)

print(len(imgs))



f = open("./results/selected_imgs_by_mcmc/selected_" + str(round(len(set(imgs)) / 27392, 2)) + "_images.txt", mode='w+',
         encoding='utf-8')
for img_name in set(imgs):
    f.write(img_name)
    f.write('\n')
f.close()
