#!/usr/bin/env python
import codecs


def dosummary(input_file, column):
	tags = ['B-PER', 'B-LOC', 'B-ORG', 'B-MISC', "SENTENCE"]
	# column = 3
	summary = {tag:0 for tag in tags}

	with codecs.open(input_file, "r", "utf8") as f:	
		for line in f.readlines():
			line = line.strip()
			if line.strip() != "":
				items = line.split("\t")
				if items[column] in summary.keys():
					summary[items[column]] = summary[items[column]] + 1
			else:
				summary["SENTENCE"] = summary["SENTENCE"] + 1
	# print summary
	for tag in tags:
		print tag + "\t" + str(summary[tag])

print "========================LAYER 1========================"
print "\tTRAIN"
dosummary("train2.conll", 3)
print "\tDEV"
dosummary("dev2.conll", 3)
print "\tTESTA"
dosummary("testa2.conll",3 )
print "\tTESTB"
dosummary("testb2.conll", 3)

print "========================LAYER 2========================"
print "\tTRAIN"
dosummary("train2.conll", 4)
print "\tDEV"
dosummary("dev2.conll", 4)
print "\tTESTA"
dosummary("testa2.conll",4 )
print "\tTESTB"
dosummary("testb2.conll", 4)