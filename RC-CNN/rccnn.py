import argparse,pwd,os,numpy as np,h5py
from os.path import splitext,exists,dirname
from os import makedirs
from itertools import izip

def outputHDF5(data,label,filename,labelname,dataname):
    print 'data shape: ',data.shape
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    label = [[x.astype(np.float32)] for x in label]
    with h5py.File(filename, 'w') as f:
    	f.create_dataset(dataname, data=data, **comp_kwargs)
    	f.create_dataset(labelname, data=label, **comp_kwargs)

def seq2feature(data,mapper,label,out_filename,worddim,labelname,dataname):
    out = []
    for seq in data:
        mat = embed(seq,mapper,worddim)
        result = mat.transpose()
        result1 = [ [a] for a in result]
        out.append(result1)
    outputHDF5(np.asarray(out),label,out_filename,labelname,dataname)

def embed(seq,mapper,worddim):
    mat = np.asarray([mapper[element] if element in mapper else np.random.rand(worddim)*2-1 for element in seq])
    return mat

def reverseComplement(sequence):
	#sequence_re = sequence[::-1]
	sequence_re = list(sequence)
	sequence_re.reverse()
	temp = sequence_re
	for index in range(len(temp)):
		if temp[index] == 'A': sequence_re[index] = 'T'
		elif temp[index] == 'C': sequence_re[index] = 'G'
		elif temp[index] == 'G': sequence_re[index] = 'C'
		elif temp[index] == 'T': sequence_re[index] = 'A'
	
	return (''.join(sequence_re))
	
def convert(infile,labelfile,outfile,mapper,worddim,kernelsize,batchsize,labelname,dataname):
    with open(infile) as seqfile, open(labelfile) as labelfile:
        cnt = 0
        seqdata = []
        label = []
        batchnum = 0
        for x,y in izip(seqfile,labelfile):
           sequence = x.strip().split()[1]
           sequence_re = reverseComplement(sequence)
           temp = list(sequence) + ['N']*kernelsize + list(sequence_re)
           seqdata.append(temp)
           label.append(float(y.strip()))
           cnt = (cnt+1)% batchsize
           if cnt == 0:
              batchnum = batchnum + 1
              seqdata = np.asarray(seqdata)
              label = np.asarray(label)
              t_outfile = outfile + '.batch' + str(batchnum)
              seq2feature(seqdata,mapper,label,t_outfile,worddim,labelname,dataname)
              seqdata = []
              label = []
        if cnt >0:
           batchnum = batchnum + 1
           seqdata = np.asarray(seqdata)
           label = np.asarray(label)
           t_outfile = outfile + '.batch' + str(batchnum)
           seq2feature(seqdata,mapper,label,t_outfile,worddim,labelname,dataname)
    return batchnum

def parse_args():
    parser = argparse.ArgumentParser(description="Convert sequence and target for Caffe")
    user = pwd.getpwuid(os.getuid())[0]

    # Positional (unnamed) arguments:
    parser.add_argument("infile",  type=str, help="Sequence in FASTA/TSV format (with .fa/.fasta or .tsv extension)")
    parser.add_argument("labelfile",  type=str,help="Label of the sequence. One number per line")
    parser.add_argument("outfile",  type=str, help="Output file (example: $your_path$/data/train.h5). ")

    # Optional arguments:
    parser.add_argument("-m", "--mapperfile", dest="mapperfile", default="", help="A TSV file mapping each nucleotide to a vector. The first column should be the nucleotide, and the rest denote the vectors. (Default mapping: A:[1,0,0,0],C:[0,1,0,0],G:[0,0,1,0],T:[0,0,0,1])")
    parser.add_argument("-k", "--kernelsize", dest="kernelsize", type=int, default=24, help="The kernel szie (motif length) of the convolutional layer")
    parser.add_argument("-b", "--batch", dest="batch", type=int, default=5000, help="Batch size for data storage (Defalt:5000)")
    parser.add_argument("-l", "--labelname", dest="labelname", default='label', help="The group name for labels in the HDF5 file")
    parser.add_argument("-d", "--dataname", dest="dataname", default='data', help="The group name for data in the HDF5 file")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    outdir = dirname(args.outfile)
    if not exists(outdir):
        makedirs(outdir)

    if args.mapperfile == "":
        args.mapper = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    else:
        args.mapper = {}
        with open(args.mapperfile,'r') as f:
            for x in f:
                line = x.strip().split()
                word = line[0]
                vec = [float(item) for item in line[1:]]
                args.mapper[word] = vec
    
    batchnum = convert(args.infile,args.labelfile,args.outfile,args.mapper,len(args.mapper['A']), \
                       args.kernelsize,args.batch,args.labelname,args.dataname)
