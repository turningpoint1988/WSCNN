import argparse,pwd,os,numpy as np,h5py,sys
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

def seq2feature(data,mapper,label,out_filename,worddim,labelname,dataname,instance_len,instance_stride,kernelsize):
    out = []
    for seq in data:
        result = embed(seq,mapper,worddim,instance_len,instance_stride,kernelsize)
        out.append(result)
    outputHDF5(np.asarray(out),label,out_filename,labelname,dataname)

def reverseComplement(sequence):
	sequence_re = sequence[::-1]
	temp = sequence_re
	for index in range(len(temp)):
		if temp[index] == 'A': sequence_re[index] = 'T'
		elif temp[index] == 'C': sequence_re[index] = 'G'
		elif temp[index] == 'G': sequence_re[index] = 'C'
		elif temp[index] == 'T': sequence_re[index] = 'A'
		else: sequence_re[index] = 'N'
	
	return (sequence_re)

def embed(seq,mapper,worddim,instance_len,instance_stride,kernelsize):
	
	instance_num = int((len(seq)-instance_len)/instance_stride) + 1
	bag = []
	for i in range(instance_num):
		instance_fw = seq[i*instance_stride:i*instance_stride+instance_len]
		if len(instance_fw) < instance_len:
			print >> sys.stderr, "the length of the instance is not right."; sys.exit(1)
		instance_bw = reverseComplement(instance_fw)
		instance = instance_fw + ['N']*kernelsize + instance_bw
		bag.append(instance)
	mat = np.asarray([mapper[element] if element in mapper else np.random.rand(worddim)*2-1 for element in bag[0]])
	mat = mat.transpose()
	result = np.asarray([ [a] for a in mat])
	for instance in bag[1:]:
		mat = np.asarray([mapper[element] if element in mapper else np.random.rand(worddim)*2-1 for element in instance])
		mat = mat.transpose()
		result1 = np.asarray([ [a] for a in mat])
		result = np.concatenate((result,result1),axis = 1)
	return result

def convert(infile,labelfile,outfile,mapper,worddim,batchsize,labelname,dataname,instance_len,instance_stride,kernelsize):
    with open(infile) as seqfile, open(labelfile) as labelfile:
        cnt = 0
        seqdata = []
        label = []
        batchnum = 0
        for x,y in izip(seqfile,labelfile):
            seqdata.append(list(x.strip().split()[1]))
            label.append(float(y.strip()))
            cnt = (cnt+1)% batchsize
            if cnt == 0:
                batchnum = batchnum + 1
                label = np.asarray(label)
                t_outfile = outfile + '.batch' + str(batchnum)
                seq2feature(seqdata,mapper,label,t_outfile,worddim,labelname,dataname,instance_len,instance_stride,kernelsize)
                seqdata = []
                label = []
        if cnt >0:
            batchnum = batchnum + 1
            label = np.asarray(label)
            t_outfile = outfile + '.batch' + str(batchnum)
            seq2feature(seqdata,mapper,label,t_outfile,worddim,labelname,dataname,instance_len,instance_stride,kernelsize)
    return batchnum

def parse_args():
    parser = argparse.ArgumentParser(description="Convert sequence and target for Caffe")
    user = pwd.getpwuid(os.getuid())[0]

    # Positional (unnamed) arguments:
    parser.add_argument("infile",  type=str, help="Sequence in FASTA/TSV format (with .fa/.fasta or .tsv extension)")
    parser.add_argument("labelfile",  type=str,help="Label of the sequence. One number per line")
    parser.add_argument("outfile",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")

    # Optional arguments:
    parser.add_argument("-m", "--mapperfile", dest="mapperfile", default="", help="A TSV file mapping each nucleotide to a vector. The first column should be the nucleotide, and the rest denote the vectors. (Default mapping: A:[1,0,0,0],C:[0,1,0,0],G:[0,0,1,0],T:[0,0,0,1])")
    parser.add_argument("-b", "--batch", dest="batch", type=int,default=5000, help="Batch size for data storage (Defalt:5000)")
    parser.add_argument("-l", "--labelname", dest="labelname",default='label', help="The group name for labels in the HDF5 file")
    parser.add_argument("-d", "--dataname", dest="dataname",default='data', help="The group name for data in the HDF5 file")
    parser.add_argument("-c", "--instance_len", dest="instance_len", type=int, default=100, help="The length of instance")
    parser.add_argument("-s", "--instance_stride", dest="instance_stride", type=int, default=20, help="The stride of every two-instances")
    parser.add_argument("-k", "--kernelsize", dest="kernelsize", type=int, default=24, help="The kernel size (motif length) of convolutional layer")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    outdir = dirname(args.outfile)
    if not exists(outdir):
        makedirs(outdir)

    if args.mapperfile == "":
        args.mapper = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
    else:
        args.mapper = {}
        with open(args.mapperfile,'r') as f:
            for x in f:
                line = x.strip().split()
                word = line[0]
                vec = [float(item) for item in line[1:]]
                args.mapper[word] = vec

    batchnum = convert(args.infile,args.labelfile,args.outfile,args.mapper,len(args.mapper['A']),args.batch,args.labelname,args.dataname,args.instance_len,args.instance_stride,args.kernelsize)
