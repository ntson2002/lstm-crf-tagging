#!/usr/bin/env python

"""
	python remove_tag.py -i testb2.conll -o testb2_notag.conll -n 2
"""
import codecs
import optparse
optparser = optparse.OptionParser()
optparser.add_option(
    "-i", "--input", default="",
    help="Input file"
)

optparser.add_option(
    "-n", "--n", default="2",
    type='int', help="Lowercase words (this will not affect character inputs)"
)

optparser.add_option(
    "-o", "--output", default="",
    help="Output file"
)

opts = optparser.parse_args()[0]
print opts 
out_lines = []
with codecs.open(opts.input, "r", "utf8") as f:	
	for line in f.readlines():
		line = line.strip()
		if line.strip() != "":
			l = line.split("\t")[:-opts.n] + ['O']*opts.n
			print "\t".join(l)
			out_lines.append("\t".join(l))		
		else:
			print 
			out_lines.append("")

with codecs.open(opts.output, "w", "utf8") as f2:
	f2.write("\n".join(out_lines))	
