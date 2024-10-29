from enum import Enum
from functools import cached_property
import itertools
import os
import pandas as pd
import numpy as np
import io
import dataclasses
from dataclasses import dataclass

class DBType(Enum):
    DBTYPE_AMINO_ACIDS = 0
    DBTYPE_NUCLEOTIDES = 1
    DBTYPE_HMM_PROFILE = 2
    DBTYPE_PROFILE_STATE_SEQ = 3
    DBTYPE_PROFILE_STATE_PROFILE = 4
    DBTYPE_ALIGNMENT_RES = 5
    DBTYPE_CLUSTER_RES = 6
    DBTYPE_PREFILTER_RES = 7
    DBTYPE_TAXONOMICAL_RESULT = 8
    DBTYPE_INDEX_DB = 9
    DBTYPE_CA3M_DB = 10
    DBTYPE_MSA_DB = 11
    DBTYPE_GENERIC_DB = 12
    DBTYPE_OMIT_FILE = 13
    DBTYPE_PREFILTER_REV_RES = 14
    DBTYPE_OFFSETDB = 15
    DBTYPE_DIRECTORY = 16
    DBTYPE_FLATFILE = 17
    DBTYPE_SEQTAXDB = 18
    DBTYPE_STDIN = 19
    DBTYPE_URI = 20

class DBCompressionType(Enum):
    UNCOMPRESSED = 0
    COMPRESSED = 1


class buffer_metaclass(type):
  def __new__(cls, name, bases, attrs):
    return super().__new__(cls, name, bases, attrs)
  def __getitem__(cls, value):
    return cls(value)

class buffer(metaclass=buffer_metaclass):
    def __init__(self, dtype=None) -> None:
        self.dtype = np.dtype(dtype)
    def __getitem__(self,dtype):
        return _buffer(dtype)
    def parse(self, fp:io.BytesIO, size:int):
        n = size // self.dtype.itemsize
        assert n*self.dtype.itemsize == size
        return np.fromfile(fp, dtype=self.dtype, count=n)

class SerializedDBIndex:
    def __init__(self, fp):
        self.size = int.from_bytes(fp.read(8),'little')
        self.dataSize = int.from_bytes(fp.read(8),'little')
        self.last_key = int.from_bytes(fp.read(4),'little')
        self.dbtype, self.dbcompressiontype = parse_dbtype(fp.read(4))
        self.max_seq_len = int.from_bytes(fp.read(4),'little')
        self.index = pd.DataFrame(np.fromfile(fp, dtype=np.uint64, count=self.size*3).astype(np.int64).reshape(self.size,3),columns=['idx','offset','length']).set_index('idx')

    @classmethod
    def parse(cls, fp:io.BytesIO, size:int):
        return cls(fp)

class SerializedDBData:
    def __init__(self, fp):
        self.fp = fp
        self.offset = self.fp.tell()

    @classmethod
    def parse(cls, fp:io.BytesIO, size:int):
        return cls(fp)

@dataclass
class Meta:
    maxSeqLen: int
    kmerSize: int
    biasCorr : bool
    adjustedKmerSize: int
    alphabetSize: int
    mask: bool
    spacedKmer: bool
    kmerThr: int
    seqType: DBType
    srcSeqType: DBType
    headers1: bool
    headers2: bool
    splits: int

    @classmethod
    def parse(cls,fp:io.BytesIO, size:int):
        ints = np.frombuffer(fp.read(size), dtype=np.int32)
        if len(ints)==11:
            targets = ['maxSeqLen', 'kmerSize', 'adjustedKmerSize', 'alphabetSize', 'mask', 'spacedKmer', None, 'seqType', 'srcSeqType', 'headers1', 'headers2']
        else:
            targets = ['maxSeqLen', 'kmerSize', 'biasCorr', 'alphabetSize', 'mask', 'spacedKmer', 'kmerThr', 'seqType', 'srcSeqType', 'headers1', 'headers2', 'splits']
        values = dict(zip(targets,ints))
        return cls(**{f.name:f.type(values[f.name]) if f.name in values else None for f in dataclasses.fields(cls)})

@dataclass
class ScoreMatrix:

    MAX_ALIGN_INT = 64
    alphabetSize : int
    kmerSize : int
    score : np.array
    index : np.array

    @classmethod
    def parse(cls, fp:io.BytesIO, bsize:int, alphabetSize:int, kmerSize:int):
        size = alphabetSize ** kmerSize
        row_size = size // cls.MAX_ALIGN_INT
        row_size = (row_size + 1) * cls.MAX_ALIGN_INT # for SIMD memory alignment
        dtypes = [np.int16, np.uint32]
        assert size * row_size * sum(np.dtype(dtype).itemsize for dtype in dtypes) == bsize

        score, index = (np.fromfile(fp, dtype=dtype, count=size*row_size).reshape(size,row_size)[:,:size] for dtype in dtypes)
        return ScoreMatrix(alphabetSize,kmerSize,score,index)


@dataclass
class BaseScoreMatrix:
    @classmethod
    def parse(cls, fp:io.BytesIO, size:int):
        df = pd.read_csv(io.StringIO('\n'.join([line for line in fp.read(size).decode().split('\n') if not line.startswith('#')][1:])), sep='\s+')
        return df

    @classmethod
    def kmers_MMorder(cls, alphabetSize:int, kmerSize:int):
        for item in itertools.product(range(alphabetSize), repeat=kmerSize):
            yield item[::-1]

    @classmethod
    def generate(cls, df:pd.DataFrame, kmerSize:int):
        # NOTE: indexes might differ due to sort-stability
        alphabetSize = len(df)-1
        scores = []
        indexes = []
        for i,kmer in enumerate(cls.kmers_MMorder(alphabetSize, kmerSize)):
            row = sorted(((sum(df.values[b1][b2] for b1,b2 in zip(kmer,kmer_)),j) for j,kmer_ in enumerate(cls.kmers_MMorder(alphabetSize, kmerSize))), reverse=True)
            score,index = zip(*row)
            scores.append(score)
            indexes.append(index)
        return ScoreMatrix(alphabetSize=alphabetSize, kmerSize=kmerSize, score=np.asarray(scores)*4, index=np.asarray(indexes))


@dataclass
class IndexEntryLocal:
    seq_id : int
    position_j : int
    @classmethod
    def parse_one(cls,fp:io.BytesIO, size:int):
        seq_id = int.from_bytes(fp.read(4),'little')
        position_j = int.from_bytes(fp.read(2),'little')
        return cls(seq_id, position_j)
    @classmethod
    def parse(cls,fp:io.BytesIO, size:int):
        return [cls.parse_one(fp, 6) for _ in range(size // 6)]

IndexEntry = np.dtype([('seq_id','u4'), ('position_j','u2')])

class PrefilteringIndex(Enum):

    VERSION : str   = 0
    META    : Meta  = 1

    SCOREMATRIXNAME : BaseScoreMatrix = 2
    SCOREMATRIX2MER : ScoreMatrix     = 3
    SCOREMATRIX3MER : ScoreMatrix     = 4

    DBR1INDEX : SerializedDBIndex = 5
    DBR1DATA  : SerializedDBData  = 6
    DBR2INDEX : SerializedDBIndex = 7
    DBR2DATA  : SerializedDBData  = 8

    ENTRIES         : buffer[IndexEntry] = 9
    ENTRIESOFFSETS  : buffer[np.uint64] = 10
    ENTRIESGRIDSIZE               = 11
    ENTRIESNUM      : int = 12

    SEQCOUNT          : int = 13
    SEQINDEXDATA      : buffer[np.uint8] = 14
    SEQINDEXDATASIZE  : int = 15
    SEQINDEXSEQOFFSET : buffer[np.uint64] = 16

    HDR1INDEX   : SerializedDBIndex = 18
    HDR1DATA    : SerializedDBData  = 19
    HDR2INDEX   : SerializedDBIndex = 20
    HDR2DATA    : SerializedDBData  = 21

    GENERATOR       : str = 22
    SPACEDPATTERN   : str = 23
    ALNINDEX = 24
    ALNDATA = 25

    @property
    def type(self):
        return type(self).__annotations__.get(self.name)

def parse_dbtype(db:int|bytes|str) -> tuple[DBType, DBCompressionType]:
    if isinstance(db,int):
        return DBType(db & 0xffff), DBCompressionType(db>>31)
    if isinstance(db,bytes):
        return parse_dbtype(int.from_bytes(db,'little'))
    if isinstance(db,str):
        return parse_dbtype(open(db+'.dbtype','rb').read())
    raise ValueError(type(db))

def get_datafiles(path:str):
    result = []
    while os.path.exists(f"{path}.{len(result)}"):
        result.append(f"{path}.{len(result)}")
    if not result and os.path.exists(path):
        result.append(path)
    return result

class MultiFp:
    def __init__(self, paths):
        self.paths = paths
        sizes = [os.stat(path).st_size for path in paths]
        self.offsets = np.cumsum(sizes)
        self.fps = [open(path,'rb') for path in self.paths]

    def seek(self, offset:int):
        idx = np.searchsorted(self.offsets, offset, side='right')
        if idx:
            offset -= self.offsets[idx-1]
        fp = self.fps[idx]
        fp.seek(offset)
        return fp

    def __repr__(self):
        return f"MultiFP({self.paths})"

class DBReader:
    def init_path(self, path) -> None:
        self._offset = 0
        self.type, self.compression_type = parse_dbtype(path)
        self.index = pd.read_csv(path+'.index',sep='\t',names=['offset','length'],index_col=0)
        # self.fp = open(path,'rb')
        self.fp = MultiFp(get_datafiles(path))

    def init_serialized(self, dbi:SerializedDBIndex, dbd:SerializedDBData):
        self._offset = dbd.offset
        self.fp = dbd.fp
        self.type = dbi.dbtype
        self.compression_type = dbi.dbcompressiontype
        self.index = dbi.index.copy()

    def __init__(self, *args) -> None:
        if len(args)==1:
            self.init_path(*args)
        else:
            self.init_serialized(*args)

    def pre_load(self, idx:int):
        row = self.index.loc[idx]
        fp = self.fp.seek(row.offset + self._offset)
        if isinstance(fp, int):
            fp = self.fp
        return fp, row.length - 1

    def load(self, idx:int):
        fp, length = self.pre_load(idx)
        data = fp.read(length)
        assert fp.read(1) == b'\x00' # check null-byte
        return data

    def close(self):
        self.fp.close()

    def __repr__(self):
        return f"<{type(self).__name__} {self.type}, {self.compression_type}, at 0x{id(self):x}>"

class IndexDBReader(DBReader):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.index['field'] = self.index.index.to_series().map(PrefilteringIndex)
        self.meta = self.load_decode(PrefilteringIndex.META)

    def load_decode(self, idx:int|PrefilteringIndex):
        if isinstance(idx,int):
            idx = PrefilteringIndex(idx)
        fp,sz = self.pre_load(idx.value)
        d = lambda:fp.read(sz)
        if idx.type is int:
            return int.from_bytes(d(), 'little')
        elif idx.type is bytes:
            return d()
        elif idx is PrefilteringIndex.SCOREMATRIX2MER:
            return ScoreMatrix.parse(fp,sz,self.meta.alphabetSize-1,2)
        elif idx is PrefilteringIndex.SCOREMATRIX3MER:
            return ScoreMatrix.parse(fp,sz,self.meta.alphabetSize-1,3)
        elif hasattr(idx.type,'parse'):
            return idx.type.parse(fp,sz)
        elif idx.type is str:
            return d().decode()
        elif idx.type is SerializedDBIndex:
            return SerializedDBIndex(fp) # Pray we don't read too much
        elif idx.type is SerializedDBData:
            return SerializedDBData(fp) # Pray we don't read too much
        elif idx.type is None:
            raise NotImplementedError(f"{idx} did not specify a type!")
        else:
            raise NotImplementedError(idx.type)


class PrefilteringDBReader(DBReader):
    def __init__(self, path) -> None:
        super().__init__(path)

    def load_decode(self, idx:int):
        fp, size = self.pre_load(idx)
        data = fp.read(size)
        return pd.read_csv(io.BytesIO(data),sep='\t',names=['targetID','ungappedScore','diagonal'],index_col=0)

def assign_property(e:PrefilteringIndex):
    name = e.name.lower()
    @cached_property
    def _getter(self:IndexDBReader):
        return self.load_decode(e)

    _getter.attrname = name

    setattr(IndexDBReader,name,_getter)


def db_open(path):
    t,_ = parse_dbtype(path)
    if t==DBType.DBTYPE_INDEX_DB:
        return IndexDBReader(path)
    if t==DBType.DBTYPE_PREFILTER_RES:
        return PrefilteringDBReader(path)
    return DBReader(path)

# for e in PrefilteringIndex:
#     assign_property(e)



# TODO: handle compressed/non-compressed
# TODO: handle db fromfile/fromotherDB (which could be compressed)
# TODO: handle dbtype specific indices
# TODO: handle splits

if False:
    db = DBReader('/tmp/queryDB')
    db = IndexDBReader('/tmp/targetDB.idx')
    dbr1 = DBReader(db.load_decode(PrefilteringIndex.DBR1INDEX), db.load_decode(PrefilteringIndex.DBR1DATA))
    dbr2 = DBReader(db.load_decode(PrefilteringIndex.DBR2INDEX), db.load_decode(PrefilteringIndex.DBR2DATA))
    dbr1.load(19996)

    m2 = db.load_decode(PrefilteringIndex.SCOREMATRIX2MER)
    df = db.load_decode(PrefilteringIndex.SCOREMATRIXNAME)
    m1 = BaseScoreMatrix.generate(df, 2)

    def idx_to_kmer(kmer_idx:int, size:int, alphabetSize:int):
        result = []
        for _ in range(size):
            result.append(kmer_idx % alphabetSize)
            kmer_idx //= alphabetSize
        return np.asarray(result)

    entries_offsets = db.load_decode(PrefilteringIndex.ENTRIESOFFSETS)
    kmer_idx = (entries_offsets[1:] - entries_offsets[:-1]).argmax()
    kmer_entries = db.load_decode(PrefilteringIndex.ENTRIES)[entries_offsets[kmer_idx]:entries_offsets[kmer_idx+1]]
    kmer_offsets = db.load_decode(PrefilteringIndex.SEQINDEXSEQOFFSET)[kmer_entries['seq_id']] + kmer_entries['position_j']
    data = db.load_decode(PrefilteringIndex.SEQINDEXDATA)
    spaced_pattern = np.asarray([1, 1, 0, 1, 0, 1, 0, 0, 1, 1],dtype=bool)
    matches = data[np.add.outer(kmer_offsets.astype(int), np.where(spaced_pattern)[0])]
    assert(matches[0] == matches).all()
    parse_dbtype(0x00090000)

db = db_open('workspace/search_tmp/7402051525033374590/pref_0')