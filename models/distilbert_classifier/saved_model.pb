��9
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( "
grad_xbool( "
grad_ybool( 
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
*
Erf
x"T
y"T"
Ttype:
2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.16.12v2.16.1-0-g5bc9d26649c8��7
�
3transformer_layer_5/feedforward_output_dense/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_5/feedforward_output_dense/kernel/*
dtype0*
shape:
��*D
shared_name53transformer_layer_5/feedforward_output_dense/kernel
�
Gtransformer_layer_5/feedforward_output_dense/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_5/feedforward_output_dense/kernel* 
_output_shapes
:
��*
dtype0
�
1transformer_layer_2/feedforward_output_dense/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_2/feedforward_output_dense/bias/*
dtype0*
shape:�*B
shared_name31transformer_layer_2/feedforward_output_dense/bias
�
Etransformer_layer_2/feedforward_output_dense/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_2/feedforward_output_dense/bias*
_output_shapes	
:�*
dtype0
�
0transformer_layer_0/feedforward_layer_norm/gammaVarHandleOp*
_output_shapes
: *A

debug_name31transformer_layer_0/feedforward_layer_norm/gamma/*
dtype0*
shape:�*A
shared_name20transformer_layer_0/feedforward_layer_norm/gamma
�
Dtransformer_layer_0/feedforward_layer_norm/gamma/Read/ReadVariableOpReadVariableOp0transformer_layer_0/feedforward_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
3transformer_layer_0/feedforward_output_dense/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_0/feedforward_output_dense/kernel/*
dtype0*
shape:
��*D
shared_name53transformer_layer_0/feedforward_output_dense/kernel
�
Gtransformer_layer_0/feedforward_output_dense/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_0/feedforward_output_dense/kernel* 
_output_shapes
:
��*
dtype0
�
5transformer_layer_4/self_attention_layer/query/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_4/self_attention_layer/query/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_4/self_attention_layer/query/kernel
�
Itransformer_layer_4/self_attention_layer/query/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_4/self_attention_layer/query/kernel*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_4/feedforward_output_dense/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_4/feedforward_output_dense/kernel/*
dtype0*
shape:
��*D
shared_name53transformer_layer_4/feedforward_output_dense/kernel
�
Gtransformer_layer_4/feedforward_output_dense/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_4/feedforward_output_dense/kernel* 
_output_shapes
:
��*
dtype0
�
2transformer_layer_1/self_attention_layer_norm/betaVarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_1/self_attention_layer_norm/beta/*
dtype0*
shape:�*C
shared_name42transformer_layer_1/self_attention_layer_norm/beta
�
Ftransformer_layer_1/self_attention_layer_norm/beta/Read/ReadVariableOpReadVariableOp2transformer_layer_1/self_attention_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
3transformer_layer_1/self_attention_layer/key/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_1/self_attention_layer/key/kernel/*
dtype0*
shape:�@*D
shared_name53transformer_layer_1/self_attention_layer/key/kernel
�
Gtransformer_layer_1/self_attention_layer/key/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_1/self_attention_layer/key/kernel*#
_output_shapes
:�@*
dtype0
�
7transformer_layer_0/feedforward_intermediate_dense/biasVarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_0/feedforward_intermediate_dense/bias/*
dtype0*
shape:�*H
shared_name97transformer_layer_0/feedforward_intermediate_dense/bias
�
Ktransformer_layer_0/feedforward_intermediate_dense/bias/Read/ReadVariableOpReadVariableOp7transformer_layer_0/feedforward_intermediate_dense/bias*
_output_shapes	
:�*
dtype0
�
3transformer_layer_3/self_attention_layer_norm/gammaVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_3/self_attention_layer_norm/gamma/*
dtype0*
shape:�*D
shared_name53transformer_layer_3/self_attention_layer_norm/gamma
�
Gtransformer_layer_3/self_attention_layer_norm/gamma/Read/ReadVariableOpReadVariableOp3transformer_layer_3/self_attention_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
>transformer_layer_1/self_attention_layer/attention_output/biasVarHandleOp*
_output_shapes
: *O

debug_nameA?transformer_layer_1/self_attention_layer/attention_output/bias/*
dtype0*
shape:�*O
shared_name@>transformer_layer_1/self_attention_layer/attention_output/bias
�
Rtransformer_layer_1/self_attention_layer/attention_output/bias/Read/ReadVariableOpReadVariableOp>transformer_layer_1/self_attention_layer/attention_output/bias*
_output_shapes	
:�*
dtype0
�
5transformer_layer_1/self_attention_layer/query/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_1/self_attention_layer/query/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_1/self_attention_layer/query/kernel
�
Itransformer_layer_1/self_attention_layer/query/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_1/self_attention_layer/query/kernel*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_0/self_attention_layer/value/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_0/self_attention_layer/value/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_0/self_attention_layer/value/bias
�
Gtransformer_layer_0/self_attention_layer/value/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_0/self_attention_layer/value/bias*
_output_shapes

:@*
dtype0
�
3transformer_layer_5/self_attention_layer_norm/gammaVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_5/self_attention_layer_norm/gamma/*
dtype0*
shape:�*D
shared_name53transformer_layer_5/self_attention_layer_norm/gamma
�
Gtransformer_layer_5/self_attention_layer_norm/gamma/Read/ReadVariableOpReadVariableOp3transformer_layer_5/self_attention_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
>transformer_layer_5/self_attention_layer/attention_output/biasVarHandleOp*
_output_shapes
: *O

debug_nameA?transformer_layer_5/self_attention_layer/attention_output/bias/*
dtype0*
shape:�*O
shared_name@>transformer_layer_5/self_attention_layer/attention_output/bias
�
Rtransformer_layer_5/self_attention_layer/attention_output/bias/Read/ReadVariableOpReadVariableOp>transformer_layer_5/self_attention_layer/attention_output/bias*
_output_shapes	
:�*
dtype0
�
3transformer_layer_3/feedforward_output_dense/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_3/feedforward_output_dense/kernel/*
dtype0*
shape:
��*D
shared_name53transformer_layer_3/feedforward_output_dense/kernel
�
Gtransformer_layer_3/feedforward_output_dense/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_3/feedforward_output_dense/kernel* 
_output_shapes
:
��*
dtype0
�
@transformer_layer_3/self_attention_layer/attention_output/kernelVarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_3/self_attention_layer/attention_output/kernel/*
dtype0*
shape:@�*Q
shared_nameB@transformer_layer_3/self_attention_layer/attention_output/kernel
�
Ttransformer_layer_3/self_attention_layer/attention_output/kernel/Read/ReadVariableOpReadVariableOp@transformer_layer_3/self_attention_layer/attention_output/kernel*#
_output_shapes
:@�*
dtype0
�
5transformer_layer_3/self_attention_layer/value/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_3/self_attention_layer/value/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_3/self_attention_layer/value/kernel
�
Itransformer_layer_3/self_attention_layer/value/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_3/self_attention_layer/value/kernel*#
_output_shapes
:�@*
dtype0
�
1transformer_layer_1/feedforward_output_dense/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_1/feedforward_output_dense/bias/*
dtype0*
shape:�*B
shared_name31transformer_layer_1/feedforward_output_dense/bias
�
Etransformer_layer_1/feedforward_output_dense/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_1/feedforward_output_dense/bias*
_output_shapes	
:�*
dtype0
�
1transformer_layer_5/self_attention_layer/key/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_5/self_attention_layer/key/bias/*
dtype0*
shape
:@*B
shared_name31transformer_layer_5/self_attention_layer/key/bias
�
Etransformer_layer_5/self_attention_layer/key/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_5/self_attention_layer/key/bias*
_output_shapes

:@*
dtype0
�
7transformer_layer_3/feedforward_intermediate_dense/biasVarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_3/feedforward_intermediate_dense/bias/*
dtype0*
shape:�*H
shared_name97transformer_layer_3/feedforward_intermediate_dense/bias
�
Ktransformer_layer_3/feedforward_intermediate_dense/bias/Read/ReadVariableOpReadVariableOp7transformer_layer_3/feedforward_intermediate_dense/bias*
_output_shapes	
:�*
dtype0
�
2transformer_layer_2/self_attention_layer_norm/betaVarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_2/self_attention_layer_norm/beta/*
dtype0*
shape:�*C
shared_name42transformer_layer_2/self_attention_layer_norm/beta
�
Ftransformer_layer_2/self_attention_layer_norm/beta/Read/ReadVariableOpReadVariableOp2transformer_layer_2/self_attention_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
7transformer_layer_5/feedforward_intermediate_dense/biasVarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_5/feedforward_intermediate_dense/bias/*
dtype0*
shape:�*H
shared_name97transformer_layer_5/feedforward_intermediate_dense/bias
�
Ktransformer_layer_5/feedforward_intermediate_dense/bias/Read/ReadVariableOpReadVariableOp7transformer_layer_5/feedforward_intermediate_dense/bias*
_output_shapes	
:�*
dtype0
�
3transformer_layer_2/feedforward_output_dense/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_2/feedforward_output_dense/kernel/*
dtype0*
shape:
��*D
shared_name53transformer_layer_2/feedforward_output_dense/kernel
�
Gtransformer_layer_2/feedforward_output_dense/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_2/feedforward_output_dense/kernel* 
_output_shapes
:
��*
dtype0
�
>transformer_layer_2/self_attention_layer/attention_output/biasVarHandleOp*
_output_shapes
: *O

debug_nameA?transformer_layer_2/self_attention_layer/attention_output/bias/*
dtype0*
shape:�*O
shared_name@>transformer_layer_2/self_attention_layer/attention_output/bias
�
Rtransformer_layer_2/self_attention_layer/attention_output/bias/Read/ReadVariableOpReadVariableOp>transformer_layer_2/self_attention_layer/attention_output/bias*
_output_shapes	
:�*
dtype0
�
3transformer_layer_1/self_attention_layer_norm/gammaVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_1/self_attention_layer_norm/gamma/*
dtype0*
shape:�*D
shared_name53transformer_layer_1/self_attention_layer_norm/gamma
�
Gtransformer_layer_1/self_attention_layer_norm/gamma/Read/ReadVariableOpReadVariableOp3transformer_layer_1/self_attention_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
9transformer_layer_0/feedforward_intermediate_dense/kernelVarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_0/feedforward_intermediate_dense/kernel/*
dtype0*
shape:
��*J
shared_name;9transformer_layer_0/feedforward_intermediate_dense/kernel
�
Mtransformer_layer_0/feedforward_intermediate_dense/kernel/Read/ReadVariableOpReadVariableOp9transformer_layer_0/feedforward_intermediate_dense/kernel* 
_output_shapes
:
��*
dtype0
�
@transformer_layer_1/self_attention_layer/attention_output/kernelVarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_1/self_attention_layer/attention_output/kernel/*
dtype0*
shape:@�*Q
shared_nameB@transformer_layer_1/self_attention_layer/attention_output/kernel
�
Ttransformer_layer_1/self_attention_layer/attention_output/kernel/Read/ReadVariableOpReadVariableOp@transformer_layer_1/self_attention_layer/attention_output/kernel*#
_output_shapes
:@�*
dtype0
�
3transformer_layer_1/self_attention_layer/value/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_1/self_attention_layer/value/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_1/self_attention_layer/value/bias
�
Gtransformer_layer_1/self_attention_layer/value/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_1/self_attention_layer/value/bias*
_output_shapes

:@*
dtype0
�
5transformer_layer_0/self_attention_layer/value/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_0/self_attention_layer/value/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_0/self_attention_layer/value/kernel
�
Itransformer_layer_0/self_attention_layer/value/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_0/self_attention_layer/value/kernel*#
_output_shapes
:�@*
dtype0
�
@transformer_layer_5/self_attention_layer/attention_output/kernelVarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_5/self_attention_layer/attention_output/kernel/*
dtype0*
shape:@�*Q
shared_nameB@transformer_layer_5/self_attention_layer/attention_output/kernel
�
Ttransformer_layer_5/self_attention_layer/attention_output/kernel/Read/ReadVariableOpReadVariableOp@transformer_layer_5/self_attention_layer/attention_output/kernel*#
_output_shapes
:@�*
dtype0
�
3transformer_layer_5/self_attention_layer/value/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_5/self_attention_layer/value/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_5/self_attention_layer/value/bias
�
Gtransformer_layer_5/self_attention_layer/value/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_5/self_attention_layer/value/bias*
_output_shapes

:@*
dtype0
�
9transformer_layer_1/feedforward_intermediate_dense/kernelVarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_1/feedforward_intermediate_dense/kernel/*
dtype0*
shape:
��*J
shared_name;9transformer_layer_1/feedforward_intermediate_dense/kernel
�
Mtransformer_layer_1/feedforward_intermediate_dense/kernel/Read/ReadVariableOpReadVariableOp9transformer_layer_1/feedforward_intermediate_dense/kernel* 
_output_shapes
:
��*
dtype0
�
1transformer_layer_0/self_attention_layer/key/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_0/self_attention_layer/key/bias/*
dtype0*
shape
:@*B
shared_name31transformer_layer_0/self_attention_layer/key/bias
�
Etransformer_layer_0/self_attention_layer/key/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_0/self_attention_layer/key/bias*
_output_shapes

:@*
dtype0
�
9transformer_layer_3/feedforward_intermediate_dense/kernelVarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_3/feedforward_intermediate_dense/kernel/*
dtype0*
shape:
��*J
shared_name;9transformer_layer_3/feedforward_intermediate_dense/kernel
�
Mtransformer_layer_3/feedforward_intermediate_dense/kernel/Read/ReadVariableOpReadVariableOp9transformer_layer_3/feedforward_intermediate_dense/kernel* 
_output_shapes
:
��*
dtype0
�
3transformer_layer_2/self_attention_layer_norm/gammaVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_2/self_attention_layer_norm/gamma/*
dtype0*
shape:�*D
shared_name53transformer_layer_2/self_attention_layer_norm/gamma
�
Gtransformer_layer_2/self_attention_layer_norm/gamma/Read/ReadVariableOpReadVariableOp3transformer_layer_2/self_attention_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
3transformer_layer_0/self_attention_layer/query/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_0/self_attention_layer/query/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_0/self_attention_layer/query/bias
�
Gtransformer_layer_0/self_attention_layer/query/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_0/self_attention_layer/query/bias*
_output_shapes

:@*
dtype0
�
9transformer_layer_5/feedforward_intermediate_dense/kernelVarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_5/feedforward_intermediate_dense/kernel/*
dtype0*
shape:
��*J
shared_name;9transformer_layer_5/feedforward_intermediate_dense/kernel
�
Mtransformer_layer_5/feedforward_intermediate_dense/kernel/Read/ReadVariableOpReadVariableOp9transformer_layer_5/feedforward_intermediate_dense/kernel* 
_output_shapes
:
��*
dtype0
�
3transformer_layer_5/self_attention_layer/query/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_5/self_attention_layer/query/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_5/self_attention_layer/query/bias
�
Gtransformer_layer_5/self_attention_layer/query/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_5/self_attention_layer/query/bias*
_output_shapes

:@*
dtype0
�
2transformer_layer_4/self_attention_layer_norm/betaVarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_4/self_attention_layer_norm/beta/*
dtype0*
shape:�*C
shared_name42transformer_layer_4/self_attention_layer_norm/beta
�
Ftransformer_layer_4/self_attention_layer_norm/beta/Read/ReadVariableOpReadVariableOp2transformer_layer_4/self_attention_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
/transformer_layer_3/feedforward_layer_norm/betaVarHandleOp*
_output_shapes
: *@

debug_name20transformer_layer_3/feedforward_layer_norm/beta/*
dtype0*
shape:�*@
shared_name1/transformer_layer_3/feedforward_layer_norm/beta
�
Ctransformer_layer_3/feedforward_layer_norm/beta/Read/ReadVariableOpReadVariableOp/transformer_layer_3/feedforward_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
@transformer_layer_2/self_attention_layer/attention_output/kernelVarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_2/self_attention_layer/attention_output/kernel/*
dtype0*
shape:@�*Q
shared_nameB@transformer_layer_2/self_attention_layer/attention_output/kernel
�
Ttransformer_layer_2/self_attention_layer/attention_output/kernel/Read/ReadVariableOpReadVariableOp@transformer_layer_2/self_attention_layer/attention_output/kernel*#
_output_shapes
:@�*
dtype0
�
3transformer_layer_2/self_attention_layer/value/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_2/self_attention_layer/value/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_2/self_attention_layer/value/bias
�
Gtransformer_layer_2/self_attention_layer/value/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_2/self_attention_layer/value/bias*
_output_shapes

:@*
dtype0
�
1transformer_layer_0/feedforward_output_dense/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_0/feedforward_output_dense/bias/*
dtype0*
shape:�*B
shared_name31transformer_layer_0/feedforward_output_dense/bias
�
Etransformer_layer_0/feedforward_output_dense/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_0/feedforward_output_dense/bias*
_output_shapes	
:�*
dtype0
�
logits/biasVarHandleOp*
_output_shapes
: *

debug_namelogits/bias/*
dtype0*
shape:*
shared_namelogits/bias
g
logits/bias/Read/ReadVariableOpReadVariableOplogits/bias*
_output_shapes
:*
dtype0
�
pooled_dense/kernelVarHandleOp*
_output_shapes
: *$

debug_namepooled_dense/kernel/*
dtype0*
shape:
��*$
shared_namepooled_dense/kernel
}
'pooled_dense/kernel/Read/ReadVariableOpReadVariableOppooled_dense/kernel* 
_output_shapes
:
��*
dtype0
�
/transformer_layer_5/feedforward_layer_norm/betaVarHandleOp*
_output_shapes
: *@

debug_name20transformer_layer_5/feedforward_layer_norm/beta/*
dtype0*
shape:�*@
shared_name1/transformer_layer_5/feedforward_layer_norm/beta
�
Ctransformer_layer_5/feedforward_layer_norm/beta/Read/ReadVariableOpReadVariableOp/transformer_layer_5/feedforward_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
>transformer_layer_4/self_attention_layer/attention_output/biasVarHandleOp*
_output_shapes
: *O

debug_nameA?transformer_layer_4/self_attention_layer/attention_output/bias/*
dtype0*
shape:�*O
shared_name@>transformer_layer_4/self_attention_layer/attention_output/bias
�
Rtransformer_layer_4/self_attention_layer/attention_output/bias/Read/ReadVariableOpReadVariableOp>transformer_layer_4/self_attention_layer/attention_output/bias*
_output_shapes	
:�*
dtype0
�
9transformer_layer_2/feedforward_intermediate_dense/kernelVarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_2/feedforward_intermediate_dense/kernel/*
dtype0*
shape:
��*J
shared_name;9transformer_layer_2/feedforward_intermediate_dense/kernel
�
Mtransformer_layer_2/feedforward_intermediate_dense/kernel/Read/ReadVariableOpReadVariableOp9transformer_layer_2/feedforward_intermediate_dense/kernel* 
_output_shapes
:
��*
dtype0
�
1transformer_layer_2/self_attention_layer/key/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_2/self_attention_layer/key/bias/*
dtype0*
shape
:@*B
shared_name31transformer_layer_2/self_attention_layer/key/bias
�
Etransformer_layer_2/self_attention_layer/key/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_2/self_attention_layer/key/bias*
_output_shapes

:@*
dtype0
�
5transformer_layer_1/self_attention_layer/value/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_1/self_attention_layer/value/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_1/self_attention_layer/value/kernel
�
Itransformer_layer_1/self_attention_layer/value/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_1/self_attention_layer/value/kernel*#
_output_shapes
:�@*
dtype0
�
5transformer_layer_5/self_attention_layer/value/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_5/self_attention_layer/value/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_5/self_attention_layer/value/kernel
�
Itransformer_layer_5/self_attention_layer/value/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_5/self_attention_layer/value/kernel*#
_output_shapes
:�@*
dtype0
�
2transformer_layer_0/self_attention_layer_norm/betaVarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_0/self_attention_layer_norm/beta/*
dtype0*
shape:�*C
shared_name42transformer_layer_0/self_attention_layer_norm/beta
�
Ftransformer_layer_0/self_attention_layer_norm/beta/Read/ReadVariableOpReadVariableOp2transformer_layer_0/self_attention_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
3transformer_layer_0/self_attention_layer/key/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_0/self_attention_layer/key/kernel/*
dtype0*
shape:�@*D
shared_name53transformer_layer_0/self_attention_layer/key/kernel
�
Gtransformer_layer_0/self_attention_layer/key/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_0/self_attention_layer/key/kernel*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_5/self_attention_layer/key/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_5/self_attention_layer/key/kernel/*
dtype0*
shape:�@*D
shared_name53transformer_layer_5/self_attention_layer/key/kernel
�
Gtransformer_layer_5/self_attention_layer/key/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_5/self_attention_layer/key/kernel*#
_output_shapes
:�@*
dtype0
�
/transformer_layer_1/feedforward_layer_norm/betaVarHandleOp*
_output_shapes
: *@

debug_name20transformer_layer_1/feedforward_layer_norm/beta/*
dtype0*
shape:�*@
shared_name1/transformer_layer_1/feedforward_layer_norm/beta
�
Ctransformer_layer_1/feedforward_layer_norm/beta/Read/ReadVariableOpReadVariableOp/transformer_layer_1/feedforward_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
>transformer_layer_0/self_attention_layer/attention_output/biasVarHandleOp*
_output_shapes
: *O

debug_nameA?transformer_layer_0/self_attention_layer/attention_output/bias/*
dtype0*
shape:�*O
shared_name@>transformer_layer_0/self_attention_layer/attention_output/bias
�
Rtransformer_layer_0/self_attention_layer/attention_output/bias/Read/ReadVariableOpReadVariableOp>transformer_layer_0/self_attention_layer/attention_output/bias*
_output_shapes	
:�*
dtype0
�
5transformer_layer_0/self_attention_layer/query/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_0/self_attention_layer/query/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_0/self_attention_layer/query/kernel
�
Itransformer_layer_0/self_attention_layer/query/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_0/self_attention_layer/query/kernel*#
_output_shapes
:�@*
dtype0
�
5transformer_layer_5/self_attention_layer/query/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_5/self_attention_layer/query/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_5/self_attention_layer/query/kernel
�
Itransformer_layer_5/self_attention_layer/query/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_5/self_attention_layer/query/kernel*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_4/self_attention_layer_norm/gammaVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_4/self_attention_layer_norm/gamma/*
dtype0*
shape:�*D
shared_name53transformer_layer_4/self_attention_layer_norm/gamma
�
Gtransformer_layer_4/self_attention_layer_norm/gamma/Read/ReadVariableOpReadVariableOp3transformer_layer_4/self_attention_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
0transformer_layer_3/feedforward_layer_norm/gammaVarHandleOp*
_output_shapes
: *A

debug_name31transformer_layer_3/feedforward_layer_norm/gamma/*
dtype0*
shape:�*A
shared_name20transformer_layer_3/feedforward_layer_norm/gamma
�
Dtransformer_layer_3/feedforward_layer_norm/gamma/Read/ReadVariableOpReadVariableOp0transformer_layer_3/feedforward_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
5transformer_layer_2/self_attention_layer/value/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_2/self_attention_layer/value/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_2/self_attention_layer/value/kernel
�
Itransformer_layer_2/self_attention_layer/value/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_2/self_attention_layer/value/kernel*#
_output_shapes
:�@*
dtype0
�
embeddings_layer_norm/gammaVarHandleOp*
_output_shapes
: *,

debug_nameembeddings_layer_norm/gamma/*
dtype0*
shape:�*,
shared_nameembeddings_layer_norm/gamma
�
/embeddings_layer_norm/gamma/Read/ReadVariableOpReadVariableOpembeddings_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
logits/kernelVarHandleOp*
_output_shapes
: *

debug_namelogits/kernel/*
dtype0*
shape:	�*
shared_namelogits/kernel
p
!logits/kernel/Read/ReadVariableOpReadVariableOplogits/kernel*
_output_shapes
:	�*
dtype0
�
0transformer_layer_5/feedforward_layer_norm/gammaVarHandleOp*
_output_shapes
: *A

debug_name31transformer_layer_5/feedforward_layer_norm/gamma/*
dtype0*
shape:�*A
shared_name20transformer_layer_5/feedforward_layer_norm/gamma
�
Dtransformer_layer_5/feedforward_layer_norm/gamma/Read/ReadVariableOpReadVariableOp0transformer_layer_5/feedforward_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
3transformer_layer_4/self_attention_layer/value/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_4/self_attention_layer/value/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_4/self_attention_layer/value/bias
�
Gtransformer_layer_4/self_attention_layer/value/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_4/self_attention_layer/value/bias*
_output_shapes

:@*
dtype0
�
7transformer_layer_2/feedforward_intermediate_dense/biasVarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_2/feedforward_intermediate_dense/bias/*
dtype0*
shape:�*H
shared_name97transformer_layer_2/feedforward_intermediate_dense/bias
�
Ktransformer_layer_2/feedforward_intermediate_dense/bias/Read/ReadVariableOpReadVariableOp7transformer_layer_2/feedforward_intermediate_dense/bias*
_output_shapes	
:�*
dtype0
�
3transformer_layer_2/self_attention_layer/key/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_2/self_attention_layer/key/kernel/*
dtype0*
shape:�@*D
shared_name53transformer_layer_2/self_attention_layer/key/kernel
�
Gtransformer_layer_2/self_attention_layer/key/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_2/self_attention_layer/key/kernel*#
_output_shapes
:�@*
dtype0
�
:token_and_position_embedding/position_embedding/embeddingsVarHandleOp*
_output_shapes
: *K

debug_name=;token_and_position_embedding/position_embedding/embeddings/*
dtype0*
shape:
��*K
shared_name<:token_and_position_embedding/position_embedding/embeddings
�
Ntoken_and_position_embedding/position_embedding/embeddings/Read/ReadVariableOpReadVariableOp:token_and_position_embedding/position_embedding/embeddings* 
_output_shapes
:
��*
dtype0
�
@transformer_layer_4/self_attention_layer/attention_output/kernelVarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_4/self_attention_layer/attention_output/kernel/*
dtype0*
shape:@�*Q
shared_nameB@transformer_layer_4/self_attention_layer/attention_output/kernel
�
Ttransformer_layer_4/self_attention_layer/attention_output/kernel/Read/ReadVariableOpReadVariableOp@transformer_layer_4/self_attention_layer/attention_output/kernel*#
_output_shapes
:@�*
dtype0
�
9transformer_layer_4/feedforward_intermediate_dense/kernelVarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_4/feedforward_intermediate_dense/kernel/*
dtype0*
shape:
��*J
shared_name;9transformer_layer_4/feedforward_intermediate_dense/kernel
�
Mtransformer_layer_4/feedforward_intermediate_dense/kernel/Read/ReadVariableOpReadVariableOp9transformer_layer_4/feedforward_intermediate_dense/kernel* 
_output_shapes
:
��*
dtype0
�
/transformer_layer_2/feedforward_layer_norm/betaVarHandleOp*
_output_shapes
: *@

debug_name20transformer_layer_2/feedforward_layer_norm/beta/*
dtype0*
shape:�*@
shared_name1/transformer_layer_2/feedforward_layer_norm/beta
�
Ctransformer_layer_2/feedforward_layer_norm/beta/Read/ReadVariableOpReadVariableOp/transformer_layer_2/feedforward_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
3transformer_layer_2/self_attention_layer/query/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_2/self_attention_layer/query/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_2/self_attention_layer/query/bias
�
Gtransformer_layer_2/self_attention_layer/query/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_2/self_attention_layer/query/bias*
_output_shapes

:@*
dtype0
�
pooled_dense/biasVarHandleOp*
_output_shapes
: *"

debug_namepooled_dense/bias/*
dtype0*
shape:�*"
shared_namepooled_dense/bias
t
%pooled_dense/bias/Read/ReadVariableOpReadVariableOppooled_dense/bias*
_output_shapes	
:�*
dtype0
�
7transformer_layer_1/feedforward_intermediate_dense/biasVarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_1/feedforward_intermediate_dense/bias/*
dtype0*
shape:�*H
shared_name97transformer_layer_1/feedforward_intermediate_dense/bias
�
Ktransformer_layer_1/feedforward_intermediate_dense/bias/Read/ReadVariableOpReadVariableOp7transformer_layer_1/feedforward_intermediate_dense/bias*
_output_shapes	
:�*
dtype0
�
0transformer_layer_1/feedforward_layer_norm/gammaVarHandleOp*
_output_shapes
: *A

debug_name31transformer_layer_1/feedforward_layer_norm/gamma/*
dtype0*
shape:�*A
shared_name20transformer_layer_1/feedforward_layer_norm/gamma
�
Dtransformer_layer_1/feedforward_layer_norm/gamma/Read/ReadVariableOpReadVariableOp0transformer_layer_1/feedforward_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
@transformer_layer_0/self_attention_layer/attention_output/kernelVarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_0/self_attention_layer/attention_output/kernel/*
dtype0*
shape:@�*Q
shared_nameB@transformer_layer_0/self_attention_layer/attention_output/kernel
�
Ttransformer_layer_0/self_attention_layer/attention_output/kernel/Read/ReadVariableOpReadVariableOp@transformer_layer_0/self_attention_layer/attention_output/kernel*#
_output_shapes
:@�*
dtype0
�
3transformer_layer_1/feedforward_output_dense/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_1/feedforward_output_dense/kernel/*
dtype0*
shape:
��*D
shared_name53transformer_layer_1/feedforward_output_dense/kernel
�
Gtransformer_layer_1/feedforward_output_dense/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_1/feedforward_output_dense/kernel* 
_output_shapes
:
��*
dtype0
�
5transformer_layer_4/self_attention_layer/value/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_4/self_attention_layer/value/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_4/self_attention_layer/value/kernel
�
Itransformer_layer_4/self_attention_layer/value/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_4/self_attention_layer/value/kernel*#
_output_shapes
:�@*
dtype0
�
1transformer_layer_4/self_attention_layer/key/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_4/self_attention_layer/key/bias/*
dtype0*
shape
:@*B
shared_name31transformer_layer_4/self_attention_layer/key/bias
�
Etransformer_layer_4/self_attention_layer/key/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_4/self_attention_layer/key/bias*
_output_shapes

:@*
dtype0
�
7transformer_layer_4/feedforward_intermediate_dense/biasVarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_4/feedforward_intermediate_dense/bias/*
dtype0*
shape:�*H
shared_name97transformer_layer_4/feedforward_intermediate_dense/bias
�
Ktransformer_layer_4/feedforward_intermediate_dense/bias/Read/ReadVariableOpReadVariableOp7transformer_layer_4/feedforward_intermediate_dense/bias*
_output_shapes	
:�*
dtype0
�
0transformer_layer_2/feedforward_layer_norm/gammaVarHandleOp*
_output_shapes
: *A

debug_name31transformer_layer_2/feedforward_layer_norm/gamma/*
dtype0*
shape:�*A
shared_name20transformer_layer_2/feedforward_layer_norm/gamma
�
Dtransformer_layer_2/feedforward_layer_norm/gamma/Read/ReadVariableOpReadVariableOp0transformer_layer_2/feedforward_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
5transformer_layer_2/self_attention_layer/query/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_2/self_attention_layer/query/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_2/self_attention_layer/query/kernel
�
Itransformer_layer_2/self_attention_layer/query/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_2/self_attention_layer/query/kernel*#
_output_shapes
:�@*
dtype0
�
embeddings_layer_norm/betaVarHandleOp*
_output_shapes
: *+

debug_nameembeddings_layer_norm/beta/*
dtype0*
shape:�*+
shared_nameembeddings_layer_norm/beta
�
.embeddings_layer_norm/beta/Read/ReadVariableOpReadVariableOpembeddings_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
/transformer_layer_4/feedforward_layer_norm/betaVarHandleOp*
_output_shapes
: *@

debug_name20transformer_layer_4/feedforward_layer_norm/beta/*
dtype0*
shape:�*@
shared_name1/transformer_layer_4/feedforward_layer_norm/beta
�
Ctransformer_layer_4/feedforward_layer_norm/beta/Read/ReadVariableOpReadVariableOp/transformer_layer_4/feedforward_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
3transformer_layer_0/self_attention_layer_norm/gammaVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_0/self_attention_layer_norm/gamma/*
dtype0*
shape:�*D
shared_name53transformer_layer_0/self_attention_layer_norm/gamma
�
Gtransformer_layer_0/self_attention_layer_norm/gamma/Read/ReadVariableOpReadVariableOp3transformer_layer_0/self_attention_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
1transformer_layer_3/self_attention_layer/key/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_3/self_attention_layer/key/bias/*
dtype0*
shape
:@*B
shared_name31transformer_layer_3/self_attention_layer/key/bias
�
Etransformer_layer_3/self_attention_layer/key/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_3/self_attention_layer/key/bias*
_output_shapes

:@*
dtype0
�
3transformer_layer_3/self_attention_layer/query/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_3/self_attention_layer/query/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_3/self_attention_layer/query/bias
�
Gtransformer_layer_3/self_attention_layer/query/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_3/self_attention_layer/query/bias*
_output_shapes

:@*
dtype0
�
1transformer_layer_5/feedforward_output_dense/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_5/feedforward_output_dense/bias/*
dtype0*
shape:�*B
shared_name31transformer_layer_5/feedforward_output_dense/bias
�
Etransformer_layer_5/feedforward_output_dense/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_5/feedforward_output_dense/bias*
_output_shapes	
:�*
dtype0
�
7token_and_position_embedding/token_embedding/embeddingsVarHandleOp*
_output_shapes
: *H

debug_name:8token_and_position_embedding/token_embedding/embeddings/*
dtype0*
shape:���*H
shared_name97token_and_position_embedding/token_embedding/embeddings
�
Ktoken_and_position_embedding/token_embedding/embeddings/Read/ReadVariableOpReadVariableOp7token_and_position_embedding/token_embedding/embeddings*!
_output_shapes
:���*
dtype0
�
3transformer_layer_4/self_attention_layer/key/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_4/self_attention_layer/key/kernel/*
dtype0*
shape:�@*D
shared_name53transformer_layer_4/self_attention_layer/key/kernel
�
Gtransformer_layer_4/self_attention_layer/key/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_4/self_attention_layer/key/kernel*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_4/self_attention_layer/query/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_4/self_attention_layer/query/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_4/self_attention_layer/query/bias
�
Gtransformer_layer_4/self_attention_layer/query/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_4/self_attention_layer/query/bias*
_output_shapes

:@*
dtype0
�
/transformer_layer_0/feedforward_layer_norm/betaVarHandleOp*
_output_shapes
: *@

debug_name20transformer_layer_0/feedforward_layer_norm/beta/*
dtype0*
shape:�*@
shared_name1/transformer_layer_0/feedforward_layer_norm/beta
�
Ctransformer_layer_0/feedforward_layer_norm/beta/Read/ReadVariableOpReadVariableOp/transformer_layer_0/feedforward_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
0transformer_layer_4/feedforward_layer_norm/gammaVarHandleOp*
_output_shapes
: *A

debug_name31transformer_layer_4/feedforward_layer_norm/gamma/*
dtype0*
shape:�*A
shared_name20transformer_layer_4/feedforward_layer_norm/gamma
�
Dtransformer_layer_4/feedforward_layer_norm/gamma/Read/ReadVariableOpReadVariableOp0transformer_layer_4/feedforward_layer_norm/gamma*
_output_shapes	
:�*
dtype0
�
1transformer_layer_4/feedforward_output_dense/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_4/feedforward_output_dense/bias/*
dtype0*
shape:�*B
shared_name31transformer_layer_4/feedforward_output_dense/bias
�
Etransformer_layer_4/feedforward_output_dense/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_4/feedforward_output_dense/bias*
_output_shapes	
:�*
dtype0
�
1transformer_layer_1/self_attention_layer/key/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_1/self_attention_layer/key/bias/*
dtype0*
shape
:@*B
shared_name31transformer_layer_1/self_attention_layer/key/bias
�
Etransformer_layer_1/self_attention_layer/key/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_1/self_attention_layer/key/bias*
_output_shapes

:@*
dtype0
�
2transformer_layer_3/self_attention_layer_norm/betaVarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_3/self_attention_layer_norm/beta/*
dtype0*
shape:�*C
shared_name42transformer_layer_3/self_attention_layer_norm/beta
�
Ftransformer_layer_3/self_attention_layer_norm/beta/Read/ReadVariableOpReadVariableOp2transformer_layer_3/self_attention_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
3transformer_layer_3/self_attention_layer/key/kernelVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_3/self_attention_layer/key/kernel/*
dtype0*
shape:�@*D
shared_name53transformer_layer_3/self_attention_layer/key/kernel
�
Gtransformer_layer_3/self_attention_layer/key/kernel/Read/ReadVariableOpReadVariableOp3transformer_layer_3/self_attention_layer/key/kernel*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_1/self_attention_layer/query/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_1/self_attention_layer/query/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_1/self_attention_layer/query/bias
�
Gtransformer_layer_1/self_attention_layer/query/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_1/self_attention_layer/query/bias*
_output_shapes

:@*
dtype0
�
2transformer_layer_5/self_attention_layer_norm/betaVarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_5/self_attention_layer_norm/beta/*
dtype0*
shape:�*C
shared_name42transformer_layer_5/self_attention_layer_norm/beta
�
Ftransformer_layer_5/self_attention_layer_norm/beta/Read/ReadVariableOpReadVariableOp2transformer_layer_5/self_attention_layer_norm/beta*
_output_shapes	
:�*
dtype0
�
1transformer_layer_3/feedforward_output_dense/biasVarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_3/feedforward_output_dense/bias/*
dtype0*
shape:�*B
shared_name31transformer_layer_3/feedforward_output_dense/bias
�
Etransformer_layer_3/feedforward_output_dense/bias/Read/ReadVariableOpReadVariableOp1transformer_layer_3/feedforward_output_dense/bias*
_output_shapes	
:�*
dtype0
�
>transformer_layer_3/self_attention_layer/attention_output/biasVarHandleOp*
_output_shapes
: *O

debug_nameA?transformer_layer_3/self_attention_layer/attention_output/bias/*
dtype0*
shape:�*O
shared_name@>transformer_layer_3/self_attention_layer/attention_output/bias
�
Rtransformer_layer_3/self_attention_layer/attention_output/bias/Read/ReadVariableOpReadVariableOp>transformer_layer_3/self_attention_layer/attention_output/bias*
_output_shapes	
:�*
dtype0
�
3transformer_layer_3/self_attention_layer/value/biasVarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_3/self_attention_layer/value/bias/*
dtype0*
shape
:@*D
shared_name53transformer_layer_3/self_attention_layer/value/bias
�
Gtransformer_layer_3/self_attention_layer/value/bias/Read/ReadVariableOpReadVariableOp3transformer_layer_3/self_attention_layer/value/bias*
_output_shapes

:@*
dtype0
�
5transformer_layer_3/self_attention_layer/query/kernelVarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_3/self_attention_layer/query/kernel/*
dtype0*
shape:�@*F
shared_name75transformer_layer_3/self_attention_layer/query/kernel
�
Itransformer_layer_3/self_attention_layer/query/kernel/Read/ReadVariableOpReadVariableOp5transformer_layer_3/self_attention_layer/query/kernel*#
_output_shapes
:�@*
dtype0
�
logits/bias_1VarHandleOp*
_output_shapes
: *

debug_namelogits/bias_1/*
dtype0*
shape:*
shared_namelogits/bias_1
k
!logits/bias_1/Read/ReadVariableOpReadVariableOplogits/bias_1*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOplogits/bias_1*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
logits/kernel_1VarHandleOp*
_output_shapes
: * 

debug_namelogits/kernel_1/*
dtype0*
shape:	�* 
shared_namelogits/kernel_1
t
#logits/kernel_1/Read/ReadVariableOpReadVariableOplogits/kernel_1*
_output_shapes
:	�*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOplogits/kernel_1*
_class
loc:@Variable_1*
_output_shapes
:	�*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:	�*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	�*
dtype0
�
seed_generator_stateVarHandleOp*
_output_shapes
: *%

debug_nameseed_generator_state/*
dtype0*
shape:*%
shared_nameseed_generator_state
y
(seed_generator_state/Read/ReadVariableOpReadVariableOpseed_generator_state*
_output_shapes
:*
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOpseed_generator_state*
_class
loc:@Variable_2*
_output_shapes
:*
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape:*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0
�
pooled_dense/bias_1VarHandleOp*
_output_shapes
: *$

debug_namepooled_dense/bias_1/*
dtype0*
shape:�*$
shared_namepooled_dense/bias_1
x
'pooled_dense/bias_1/Read/ReadVariableOpReadVariableOppooled_dense/bias_1*
_output_shapes	
:�*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOppooled_dense/bias_1*
_class
loc:@Variable_3*
_output_shapes	
:�*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:�*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
f
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes	
:�*
dtype0
�
pooled_dense/kernel_1VarHandleOp*
_output_shapes
: *&

debug_namepooled_dense/kernel_1/*
dtype0*
shape:
��*&
shared_namepooled_dense/kernel_1
�
)pooled_dense/kernel_1/Read/ReadVariableOpReadVariableOppooled_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOppooled_dense/kernel_1*
_class
loc:@Variable_4* 
_output_shapes
:
��*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:
��*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
k
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4* 
_output_shapes
:
��*
dtype0
�
(transformer_layer_5/seed_generator_stateVarHandleOp*
_output_shapes
: *9

debug_name+)transformer_layer_5/seed_generator_state/*
dtype0*
shape:*9
shared_name*(transformer_layer_5/seed_generator_state
�
<transformer_layer_5/seed_generator_state/Read/ReadVariableOpReadVariableOp(transformer_layer_5/seed_generator_state*
_output_shapes
:*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOp(transformer_layer_5/seed_generator_state*
_class
loc:@Variable_5*
_output_shapes
:*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
e
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
:*
dtype0
�
3transformer_layer_5/feedforward_output_dense/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_5/feedforward_output_dense/bias_1/*
dtype0*
shape:�*D
shared_name53transformer_layer_5/feedforward_output_dense/bias_1
�
Gtransformer_layer_5/feedforward_output_dense/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_5/feedforward_output_dense/bias_1*
_output_shapes	
:�*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOp3transformer_layer_5/feedforward_output_dense/bias_1*
_class
loc:@Variable_6*
_output_shapes	
:�*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:�*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
f
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes	
:�*
dtype0
�
5transformer_layer_5/feedforward_output_dense/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_5/feedforward_output_dense/kernel_1/*
dtype0*
shape:
��*F
shared_name75transformer_layer_5/feedforward_output_dense/kernel_1
�
Itransformer_layer_5/feedforward_output_dense/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_5/feedforward_output_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOp5transformer_layer_5/feedforward_output_dense/kernel_1*
_class
loc:@Variable_7* 
_output_shapes
:
��*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:
��*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
k
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7* 
_output_shapes
:
��*
dtype0
�
9transformer_layer_5/feedforward_intermediate_dense/bias_1VarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_5/feedforward_intermediate_dense/bias_1/*
dtype0*
shape:�*J
shared_name;9transformer_layer_5/feedforward_intermediate_dense/bias_1
�
Mtransformer_layer_5/feedforward_intermediate_dense/bias_1/Read/ReadVariableOpReadVariableOp9transformer_layer_5/feedforward_intermediate_dense/bias_1*
_output_shapes	
:�*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOp9transformer_layer_5/feedforward_intermediate_dense/bias_1*
_class
loc:@Variable_8*
_output_shapes	
:�*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape:�*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
f
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes	
:�*
dtype0
�
;transformer_layer_5/feedforward_intermediate_dense/kernel_1VarHandleOp*
_output_shapes
: *L

debug_name><transformer_layer_5/feedforward_intermediate_dense/kernel_1/*
dtype0*
shape:
��*L
shared_name=;transformer_layer_5/feedforward_intermediate_dense/kernel_1
�
Otransformer_layer_5/feedforward_intermediate_dense/kernel_1/Read/ReadVariableOpReadVariableOp;transformer_layer_5/feedforward_intermediate_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOp;transformer_layer_5/feedforward_intermediate_dense/kernel_1*
_class
loc:@Variable_9* 
_output_shapes
:
��*
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape:
��*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
k
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9* 
_output_shapes
:
��*
dtype0
�
1transformer_layer_5/feedforward_layer_norm/beta_1VarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_5/feedforward_layer_norm/beta_1/*
dtype0*
shape:�*B
shared_name31transformer_layer_5/feedforward_layer_norm/beta_1
�
Etransformer_layer_5/feedforward_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp1transformer_layer_5/feedforward_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOp1transformer_layer_5/feedforward_layer_norm/beta_1*
_class
loc:@Variable_10*
_output_shapes	
:�*
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape:�*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
h
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes	
:�*
dtype0
�
2transformer_layer_5/feedforward_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_5/feedforward_layer_norm/gamma_1/*
dtype0*
shape:�*C
shared_name42transformer_layer_5/feedforward_layer_norm/gamma_1
�
Ftransformer_layer_5/feedforward_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp2transformer_layer_5/feedforward_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOp2transformer_layer_5/feedforward_layer_norm/gamma_1*
_class
loc:@Variable_11*
_output_shapes	
:�*
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape:�*
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
h
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes	
:�*
dtype0
�
*transformer_layer_5/seed_generator_state_1VarHandleOp*
_output_shapes
: *;

debug_name-+transformer_layer_5/seed_generator_state_1/*
dtype0*
shape:*;
shared_name,*transformer_layer_5/seed_generator_state_1
�
>transformer_layer_5/seed_generator_state_1/Read/ReadVariableOpReadVariableOp*transformer_layer_5/seed_generator_state_1*
_output_shapes
:*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOp*transformer_layer_5/seed_generator_state_1*
_class
loc:@Variable_12*
_output_shapes
:*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape:*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
g
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes
:*
dtype0
�
4transformer_layer_5/self_attention_layer_norm/beta_1VarHandleOp*
_output_shapes
: *E

debug_name75transformer_layer_5/self_attention_layer_norm/beta_1/*
dtype0*
shape:�*E
shared_name64transformer_layer_5/self_attention_layer_norm/beta_1
�
Htransformer_layer_5/self_attention_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp4transformer_layer_5/self_attention_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOp4transformer_layer_5/self_attention_layer_norm/beta_1*
_class
loc:@Variable_13*
_output_shapes	
:�*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:�*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
h
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes	
:�*
dtype0
�
5transformer_layer_5/self_attention_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_5/self_attention_layer_norm/gamma_1/*
dtype0*
shape:�*F
shared_name75transformer_layer_5/self_attention_layer_norm/gamma_1
�
Itransformer_layer_5/self_attention_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp5transformer_layer_5/self_attention_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOp5transformer_layer_5/self_attention_layer_norm/gamma_1*
_class
loc:@Variable_14*
_output_shapes	
:�*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape:�*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
h
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes	
:�*
dtype0
�
@transformer_layer_5/self_attention_layer/attention_output/bias_1VarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_5/self_attention_layer/attention_output/bias_1/*
dtype0*
shape:�*Q
shared_nameB@transformer_layer_5/self_attention_layer/attention_output/bias_1
�
Ttransformer_layer_5/self_attention_layer/attention_output/bias_1/Read/ReadVariableOpReadVariableOp@transformer_layer_5/self_attention_layer/attention_output/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOp@transformer_layer_5/self_attention_layer/attention_output/bias_1*
_class
loc:@Variable_15*
_output_shapes	
:�*
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape:�*
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
h
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*
_output_shapes	
:�*
dtype0
�
Btransformer_layer_5/self_attention_layer/attention_output/kernel_1VarHandleOp*
_output_shapes
: *S

debug_nameECtransformer_layer_5/self_attention_layer/attention_output/kernel_1/*
dtype0*
shape:@�*S
shared_nameDBtransformer_layer_5/self_attention_layer/attention_output/kernel_1
�
Vtransformer_layer_5/self_attention_layer/attention_output/kernel_1/Read/ReadVariableOpReadVariableOpBtransformer_layer_5/self_attention_layer/attention_output/kernel_1*#
_output_shapes
:@�*
dtype0
�
&Variable_16/Initializer/ReadVariableOpReadVariableOpBtransformer_layer_5/self_attention_layer/attention_output/kernel_1*
_class
loc:@Variable_16*#
_output_shapes
:@�*
dtype0
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0*
shape:@�*
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0
p
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*#
_output_shapes
:@�*
dtype0
�
=transformer_layer_5/self_attention_layer/seed_generator_stateVarHandleOp*
_output_shapes
: *N

debug_name@>transformer_layer_5/self_attention_layer/seed_generator_state/*
dtype0*
shape:*N
shared_name?=transformer_layer_5/self_attention_layer/seed_generator_state
�
Qtransformer_layer_5/self_attention_layer/seed_generator_state/Read/ReadVariableOpReadVariableOp=transformer_layer_5/self_attention_layer/seed_generator_state*
_output_shapes
:*
dtype0
�
&Variable_17/Initializer/ReadVariableOpReadVariableOp=transformer_layer_5/self_attention_layer/seed_generator_state*
_class
loc:@Variable_17*
_output_shapes
:*
dtype0
�
Variable_17VarHandleOp*
_class
loc:@Variable_17*
_output_shapes
: *

debug_nameVariable_17/*
dtype0*
shape:*
shared_nameVariable_17
g
,Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_17*
_output_shapes
: 
h
Variable_17/AssignAssignVariableOpVariable_17&Variable_17/Initializer/ReadVariableOp*
dtype0
g
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17*
_output_shapes
:*
dtype0
�
5transformer_layer_5/self_attention_layer/value/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_5/self_attention_layer/value/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_5/self_attention_layer/value/bias_1
�
Itransformer_layer_5/self_attention_layer/value/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_5/self_attention_layer/value/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_18/Initializer/ReadVariableOpReadVariableOp5transformer_layer_5/self_attention_layer/value/bias_1*
_class
loc:@Variable_18*
_output_shapes

:@*
dtype0
�
Variable_18VarHandleOp*
_class
loc:@Variable_18*
_output_shapes
: *

debug_nameVariable_18/*
dtype0*
shape
:@*
shared_nameVariable_18
g
,Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_18*
_output_shapes
: 
h
Variable_18/AssignAssignVariableOpVariable_18&Variable_18/Initializer/ReadVariableOp*
dtype0
k
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18*
_output_shapes

:@*
dtype0
�
7transformer_layer_5/self_attention_layer/value/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_5/self_attention_layer/value/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_5/self_attention_layer/value/kernel_1
�
Ktransformer_layer_5/self_attention_layer/value/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_5/self_attention_layer/value/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_19/Initializer/ReadVariableOpReadVariableOp7transformer_layer_5/self_attention_layer/value/kernel_1*
_class
loc:@Variable_19*#
_output_shapes
:�@*
dtype0
�
Variable_19VarHandleOp*
_class
loc:@Variable_19*
_output_shapes
: *

debug_nameVariable_19/*
dtype0*
shape:�@*
shared_nameVariable_19
g
,Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_19*
_output_shapes
: 
h
Variable_19/AssignAssignVariableOpVariable_19&Variable_19/Initializer/ReadVariableOp*
dtype0
p
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_5/self_attention_layer/key/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_5/self_attention_layer/key/bias_1/*
dtype0*
shape
:@*D
shared_name53transformer_layer_5/self_attention_layer/key/bias_1
�
Gtransformer_layer_5/self_attention_layer/key/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_5/self_attention_layer/key/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_20/Initializer/ReadVariableOpReadVariableOp3transformer_layer_5/self_attention_layer/key/bias_1*
_class
loc:@Variable_20*
_output_shapes

:@*
dtype0
�
Variable_20VarHandleOp*
_class
loc:@Variable_20*
_output_shapes
: *

debug_nameVariable_20/*
dtype0*
shape
:@*
shared_nameVariable_20
g
,Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_20*
_output_shapes
: 
h
Variable_20/AssignAssignVariableOpVariable_20&Variable_20/Initializer/ReadVariableOp*
dtype0
k
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20*
_output_shapes

:@*
dtype0
�
5transformer_layer_5/self_attention_layer/key/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_5/self_attention_layer/key/kernel_1/*
dtype0*
shape:�@*F
shared_name75transformer_layer_5/self_attention_layer/key/kernel_1
�
Itransformer_layer_5/self_attention_layer/key/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_5/self_attention_layer/key/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_21/Initializer/ReadVariableOpReadVariableOp5transformer_layer_5/self_attention_layer/key/kernel_1*
_class
loc:@Variable_21*#
_output_shapes
:�@*
dtype0
�
Variable_21VarHandleOp*
_class
loc:@Variable_21*
_output_shapes
: *

debug_nameVariable_21/*
dtype0*
shape:�@*
shared_nameVariable_21
g
,Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_21*
_output_shapes
: 
h
Variable_21/AssignAssignVariableOpVariable_21&Variable_21/Initializer/ReadVariableOp*
dtype0
p
Variable_21/Read/ReadVariableOpReadVariableOpVariable_21*#
_output_shapes
:�@*
dtype0
�
5transformer_layer_5/self_attention_layer/query/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_5/self_attention_layer/query/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_5/self_attention_layer/query/bias_1
�
Itransformer_layer_5/self_attention_layer/query/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_5/self_attention_layer/query/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_22/Initializer/ReadVariableOpReadVariableOp5transformer_layer_5/self_attention_layer/query/bias_1*
_class
loc:@Variable_22*
_output_shapes

:@*
dtype0
�
Variable_22VarHandleOp*
_class
loc:@Variable_22*
_output_shapes
: *

debug_nameVariable_22/*
dtype0*
shape
:@*
shared_nameVariable_22
g
,Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_22*
_output_shapes
: 
h
Variable_22/AssignAssignVariableOpVariable_22&Variable_22/Initializer/ReadVariableOp*
dtype0
k
Variable_22/Read/ReadVariableOpReadVariableOpVariable_22*
_output_shapes

:@*
dtype0
�
7transformer_layer_5/self_attention_layer/query/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_5/self_attention_layer/query/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_5/self_attention_layer/query/kernel_1
�
Ktransformer_layer_5/self_attention_layer/query/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_5/self_attention_layer/query/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_23/Initializer/ReadVariableOpReadVariableOp7transformer_layer_5/self_attention_layer/query/kernel_1*
_class
loc:@Variable_23*#
_output_shapes
:�@*
dtype0
�
Variable_23VarHandleOp*
_class
loc:@Variable_23*
_output_shapes
: *

debug_nameVariable_23/*
dtype0*
shape:�@*
shared_nameVariable_23
g
,Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_23*
_output_shapes
: 
h
Variable_23/AssignAssignVariableOpVariable_23&Variable_23/Initializer/ReadVariableOp*
dtype0
p
Variable_23/Read/ReadVariableOpReadVariableOpVariable_23*#
_output_shapes
:�@*
dtype0
�
(transformer_layer_4/seed_generator_stateVarHandleOp*
_output_shapes
: *9

debug_name+)transformer_layer_4/seed_generator_state/*
dtype0*
shape:*9
shared_name*(transformer_layer_4/seed_generator_state
�
<transformer_layer_4/seed_generator_state/Read/ReadVariableOpReadVariableOp(transformer_layer_4/seed_generator_state*
_output_shapes
:*
dtype0
�
&Variable_24/Initializer/ReadVariableOpReadVariableOp(transformer_layer_4/seed_generator_state*
_class
loc:@Variable_24*
_output_shapes
:*
dtype0
�
Variable_24VarHandleOp*
_class
loc:@Variable_24*
_output_shapes
: *

debug_nameVariable_24/*
dtype0*
shape:*
shared_nameVariable_24
g
,Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_24*
_output_shapes
: 
h
Variable_24/AssignAssignVariableOpVariable_24&Variable_24/Initializer/ReadVariableOp*
dtype0
g
Variable_24/Read/ReadVariableOpReadVariableOpVariable_24*
_output_shapes
:*
dtype0
�
3transformer_layer_4/feedforward_output_dense/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_4/feedforward_output_dense/bias_1/*
dtype0*
shape:�*D
shared_name53transformer_layer_4/feedforward_output_dense/bias_1
�
Gtransformer_layer_4/feedforward_output_dense/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_4/feedforward_output_dense/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_25/Initializer/ReadVariableOpReadVariableOp3transformer_layer_4/feedforward_output_dense/bias_1*
_class
loc:@Variable_25*
_output_shapes	
:�*
dtype0
�
Variable_25VarHandleOp*
_class
loc:@Variable_25*
_output_shapes
: *

debug_nameVariable_25/*
dtype0*
shape:�*
shared_nameVariable_25
g
,Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_25*
_output_shapes
: 
h
Variable_25/AssignAssignVariableOpVariable_25&Variable_25/Initializer/ReadVariableOp*
dtype0
h
Variable_25/Read/ReadVariableOpReadVariableOpVariable_25*
_output_shapes	
:�*
dtype0
�
5transformer_layer_4/feedforward_output_dense/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_4/feedforward_output_dense/kernel_1/*
dtype0*
shape:
��*F
shared_name75transformer_layer_4/feedforward_output_dense/kernel_1
�
Itransformer_layer_4/feedforward_output_dense/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_4/feedforward_output_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_26/Initializer/ReadVariableOpReadVariableOp5transformer_layer_4/feedforward_output_dense/kernel_1*
_class
loc:@Variable_26* 
_output_shapes
:
��*
dtype0
�
Variable_26VarHandleOp*
_class
loc:@Variable_26*
_output_shapes
: *

debug_nameVariable_26/*
dtype0*
shape:
��*
shared_nameVariable_26
g
,Variable_26/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_26*
_output_shapes
: 
h
Variable_26/AssignAssignVariableOpVariable_26&Variable_26/Initializer/ReadVariableOp*
dtype0
m
Variable_26/Read/ReadVariableOpReadVariableOpVariable_26* 
_output_shapes
:
��*
dtype0
�
9transformer_layer_4/feedforward_intermediate_dense/bias_1VarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_4/feedforward_intermediate_dense/bias_1/*
dtype0*
shape:�*J
shared_name;9transformer_layer_4/feedforward_intermediate_dense/bias_1
�
Mtransformer_layer_4/feedforward_intermediate_dense/bias_1/Read/ReadVariableOpReadVariableOp9transformer_layer_4/feedforward_intermediate_dense/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_27/Initializer/ReadVariableOpReadVariableOp9transformer_layer_4/feedforward_intermediate_dense/bias_1*
_class
loc:@Variable_27*
_output_shapes	
:�*
dtype0
�
Variable_27VarHandleOp*
_class
loc:@Variable_27*
_output_shapes
: *

debug_nameVariable_27/*
dtype0*
shape:�*
shared_nameVariable_27
g
,Variable_27/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_27*
_output_shapes
: 
h
Variable_27/AssignAssignVariableOpVariable_27&Variable_27/Initializer/ReadVariableOp*
dtype0
h
Variable_27/Read/ReadVariableOpReadVariableOpVariable_27*
_output_shapes	
:�*
dtype0
�
;transformer_layer_4/feedforward_intermediate_dense/kernel_1VarHandleOp*
_output_shapes
: *L

debug_name><transformer_layer_4/feedforward_intermediate_dense/kernel_1/*
dtype0*
shape:
��*L
shared_name=;transformer_layer_4/feedforward_intermediate_dense/kernel_1
�
Otransformer_layer_4/feedforward_intermediate_dense/kernel_1/Read/ReadVariableOpReadVariableOp;transformer_layer_4/feedforward_intermediate_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_28/Initializer/ReadVariableOpReadVariableOp;transformer_layer_4/feedforward_intermediate_dense/kernel_1*
_class
loc:@Variable_28* 
_output_shapes
:
��*
dtype0
�
Variable_28VarHandleOp*
_class
loc:@Variable_28*
_output_shapes
: *

debug_nameVariable_28/*
dtype0*
shape:
��*
shared_nameVariable_28
g
,Variable_28/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_28*
_output_shapes
: 
h
Variable_28/AssignAssignVariableOpVariable_28&Variable_28/Initializer/ReadVariableOp*
dtype0
m
Variable_28/Read/ReadVariableOpReadVariableOpVariable_28* 
_output_shapes
:
��*
dtype0
�
1transformer_layer_4/feedforward_layer_norm/beta_1VarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_4/feedforward_layer_norm/beta_1/*
dtype0*
shape:�*B
shared_name31transformer_layer_4/feedforward_layer_norm/beta_1
�
Etransformer_layer_4/feedforward_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp1transformer_layer_4/feedforward_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_29/Initializer/ReadVariableOpReadVariableOp1transformer_layer_4/feedforward_layer_norm/beta_1*
_class
loc:@Variable_29*
_output_shapes	
:�*
dtype0
�
Variable_29VarHandleOp*
_class
loc:@Variable_29*
_output_shapes
: *

debug_nameVariable_29/*
dtype0*
shape:�*
shared_nameVariable_29
g
,Variable_29/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_29*
_output_shapes
: 
h
Variable_29/AssignAssignVariableOpVariable_29&Variable_29/Initializer/ReadVariableOp*
dtype0
h
Variable_29/Read/ReadVariableOpReadVariableOpVariable_29*
_output_shapes	
:�*
dtype0
�
2transformer_layer_4/feedforward_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_4/feedforward_layer_norm/gamma_1/*
dtype0*
shape:�*C
shared_name42transformer_layer_4/feedforward_layer_norm/gamma_1
�
Ftransformer_layer_4/feedforward_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp2transformer_layer_4/feedforward_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_30/Initializer/ReadVariableOpReadVariableOp2transformer_layer_4/feedforward_layer_norm/gamma_1*
_class
loc:@Variable_30*
_output_shapes	
:�*
dtype0
�
Variable_30VarHandleOp*
_class
loc:@Variable_30*
_output_shapes
: *

debug_nameVariable_30/*
dtype0*
shape:�*
shared_nameVariable_30
g
,Variable_30/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_30*
_output_shapes
: 
h
Variable_30/AssignAssignVariableOpVariable_30&Variable_30/Initializer/ReadVariableOp*
dtype0
h
Variable_30/Read/ReadVariableOpReadVariableOpVariable_30*
_output_shapes	
:�*
dtype0
�
*transformer_layer_4/seed_generator_state_1VarHandleOp*
_output_shapes
: *;

debug_name-+transformer_layer_4/seed_generator_state_1/*
dtype0*
shape:*;
shared_name,*transformer_layer_4/seed_generator_state_1
�
>transformer_layer_4/seed_generator_state_1/Read/ReadVariableOpReadVariableOp*transformer_layer_4/seed_generator_state_1*
_output_shapes
:*
dtype0
�
&Variable_31/Initializer/ReadVariableOpReadVariableOp*transformer_layer_4/seed_generator_state_1*
_class
loc:@Variable_31*
_output_shapes
:*
dtype0
�
Variable_31VarHandleOp*
_class
loc:@Variable_31*
_output_shapes
: *

debug_nameVariable_31/*
dtype0*
shape:*
shared_nameVariable_31
g
,Variable_31/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_31*
_output_shapes
: 
h
Variable_31/AssignAssignVariableOpVariable_31&Variable_31/Initializer/ReadVariableOp*
dtype0
g
Variable_31/Read/ReadVariableOpReadVariableOpVariable_31*
_output_shapes
:*
dtype0
�
4transformer_layer_4/self_attention_layer_norm/beta_1VarHandleOp*
_output_shapes
: *E

debug_name75transformer_layer_4/self_attention_layer_norm/beta_1/*
dtype0*
shape:�*E
shared_name64transformer_layer_4/self_attention_layer_norm/beta_1
�
Htransformer_layer_4/self_attention_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp4transformer_layer_4/self_attention_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_32/Initializer/ReadVariableOpReadVariableOp4transformer_layer_4/self_attention_layer_norm/beta_1*
_class
loc:@Variable_32*
_output_shapes	
:�*
dtype0
�
Variable_32VarHandleOp*
_class
loc:@Variable_32*
_output_shapes
: *

debug_nameVariable_32/*
dtype0*
shape:�*
shared_nameVariable_32
g
,Variable_32/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_32*
_output_shapes
: 
h
Variable_32/AssignAssignVariableOpVariable_32&Variable_32/Initializer/ReadVariableOp*
dtype0
h
Variable_32/Read/ReadVariableOpReadVariableOpVariable_32*
_output_shapes	
:�*
dtype0
�
5transformer_layer_4/self_attention_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_4/self_attention_layer_norm/gamma_1/*
dtype0*
shape:�*F
shared_name75transformer_layer_4/self_attention_layer_norm/gamma_1
�
Itransformer_layer_4/self_attention_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp5transformer_layer_4/self_attention_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_33/Initializer/ReadVariableOpReadVariableOp5transformer_layer_4/self_attention_layer_norm/gamma_1*
_class
loc:@Variable_33*
_output_shapes	
:�*
dtype0
�
Variable_33VarHandleOp*
_class
loc:@Variable_33*
_output_shapes
: *

debug_nameVariable_33/*
dtype0*
shape:�*
shared_nameVariable_33
g
,Variable_33/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_33*
_output_shapes
: 
h
Variable_33/AssignAssignVariableOpVariable_33&Variable_33/Initializer/ReadVariableOp*
dtype0
h
Variable_33/Read/ReadVariableOpReadVariableOpVariable_33*
_output_shapes	
:�*
dtype0
�
@transformer_layer_4/self_attention_layer/attention_output/bias_1VarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_4/self_attention_layer/attention_output/bias_1/*
dtype0*
shape:�*Q
shared_nameB@transformer_layer_4/self_attention_layer/attention_output/bias_1
�
Ttransformer_layer_4/self_attention_layer/attention_output/bias_1/Read/ReadVariableOpReadVariableOp@transformer_layer_4/self_attention_layer/attention_output/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_34/Initializer/ReadVariableOpReadVariableOp@transformer_layer_4/self_attention_layer/attention_output/bias_1*
_class
loc:@Variable_34*
_output_shapes	
:�*
dtype0
�
Variable_34VarHandleOp*
_class
loc:@Variable_34*
_output_shapes
: *

debug_nameVariable_34/*
dtype0*
shape:�*
shared_nameVariable_34
g
,Variable_34/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_34*
_output_shapes
: 
h
Variable_34/AssignAssignVariableOpVariable_34&Variable_34/Initializer/ReadVariableOp*
dtype0
h
Variable_34/Read/ReadVariableOpReadVariableOpVariable_34*
_output_shapes	
:�*
dtype0
�
Btransformer_layer_4/self_attention_layer/attention_output/kernel_1VarHandleOp*
_output_shapes
: *S

debug_nameECtransformer_layer_4/self_attention_layer/attention_output/kernel_1/*
dtype0*
shape:@�*S
shared_nameDBtransformer_layer_4/self_attention_layer/attention_output/kernel_1
�
Vtransformer_layer_4/self_attention_layer/attention_output/kernel_1/Read/ReadVariableOpReadVariableOpBtransformer_layer_4/self_attention_layer/attention_output/kernel_1*#
_output_shapes
:@�*
dtype0
�
&Variable_35/Initializer/ReadVariableOpReadVariableOpBtransformer_layer_4/self_attention_layer/attention_output/kernel_1*
_class
loc:@Variable_35*#
_output_shapes
:@�*
dtype0
�
Variable_35VarHandleOp*
_class
loc:@Variable_35*
_output_shapes
: *

debug_nameVariable_35/*
dtype0*
shape:@�*
shared_nameVariable_35
g
,Variable_35/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_35*
_output_shapes
: 
h
Variable_35/AssignAssignVariableOpVariable_35&Variable_35/Initializer/ReadVariableOp*
dtype0
p
Variable_35/Read/ReadVariableOpReadVariableOpVariable_35*#
_output_shapes
:@�*
dtype0
�
=transformer_layer_4/self_attention_layer/seed_generator_stateVarHandleOp*
_output_shapes
: *N

debug_name@>transformer_layer_4/self_attention_layer/seed_generator_state/*
dtype0*
shape:*N
shared_name?=transformer_layer_4/self_attention_layer/seed_generator_state
�
Qtransformer_layer_4/self_attention_layer/seed_generator_state/Read/ReadVariableOpReadVariableOp=transformer_layer_4/self_attention_layer/seed_generator_state*
_output_shapes
:*
dtype0
�
&Variable_36/Initializer/ReadVariableOpReadVariableOp=transformer_layer_4/self_attention_layer/seed_generator_state*
_class
loc:@Variable_36*
_output_shapes
:*
dtype0
�
Variable_36VarHandleOp*
_class
loc:@Variable_36*
_output_shapes
: *

debug_nameVariable_36/*
dtype0*
shape:*
shared_nameVariable_36
g
,Variable_36/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_36*
_output_shapes
: 
h
Variable_36/AssignAssignVariableOpVariable_36&Variable_36/Initializer/ReadVariableOp*
dtype0
g
Variable_36/Read/ReadVariableOpReadVariableOpVariable_36*
_output_shapes
:*
dtype0
�
5transformer_layer_4/self_attention_layer/value/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_4/self_attention_layer/value/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_4/self_attention_layer/value/bias_1
�
Itransformer_layer_4/self_attention_layer/value/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_4/self_attention_layer/value/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_37/Initializer/ReadVariableOpReadVariableOp5transformer_layer_4/self_attention_layer/value/bias_1*
_class
loc:@Variable_37*
_output_shapes

:@*
dtype0
�
Variable_37VarHandleOp*
_class
loc:@Variable_37*
_output_shapes
: *

debug_nameVariable_37/*
dtype0*
shape
:@*
shared_nameVariable_37
g
,Variable_37/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_37*
_output_shapes
: 
h
Variable_37/AssignAssignVariableOpVariable_37&Variable_37/Initializer/ReadVariableOp*
dtype0
k
Variable_37/Read/ReadVariableOpReadVariableOpVariable_37*
_output_shapes

:@*
dtype0
�
7transformer_layer_4/self_attention_layer/value/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_4/self_attention_layer/value/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_4/self_attention_layer/value/kernel_1
�
Ktransformer_layer_4/self_attention_layer/value/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_4/self_attention_layer/value/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_38/Initializer/ReadVariableOpReadVariableOp7transformer_layer_4/self_attention_layer/value/kernel_1*
_class
loc:@Variable_38*#
_output_shapes
:�@*
dtype0
�
Variable_38VarHandleOp*
_class
loc:@Variable_38*
_output_shapes
: *

debug_nameVariable_38/*
dtype0*
shape:�@*
shared_nameVariable_38
g
,Variable_38/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_38*
_output_shapes
: 
h
Variable_38/AssignAssignVariableOpVariable_38&Variable_38/Initializer/ReadVariableOp*
dtype0
p
Variable_38/Read/ReadVariableOpReadVariableOpVariable_38*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_4/self_attention_layer/key/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_4/self_attention_layer/key/bias_1/*
dtype0*
shape
:@*D
shared_name53transformer_layer_4/self_attention_layer/key/bias_1
�
Gtransformer_layer_4/self_attention_layer/key/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_4/self_attention_layer/key/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_39/Initializer/ReadVariableOpReadVariableOp3transformer_layer_4/self_attention_layer/key/bias_1*
_class
loc:@Variable_39*
_output_shapes

:@*
dtype0
�
Variable_39VarHandleOp*
_class
loc:@Variable_39*
_output_shapes
: *

debug_nameVariable_39/*
dtype0*
shape
:@*
shared_nameVariable_39
g
,Variable_39/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_39*
_output_shapes
: 
h
Variable_39/AssignAssignVariableOpVariable_39&Variable_39/Initializer/ReadVariableOp*
dtype0
k
Variable_39/Read/ReadVariableOpReadVariableOpVariable_39*
_output_shapes

:@*
dtype0
�
5transformer_layer_4/self_attention_layer/key/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_4/self_attention_layer/key/kernel_1/*
dtype0*
shape:�@*F
shared_name75transformer_layer_4/self_attention_layer/key/kernel_1
�
Itransformer_layer_4/self_attention_layer/key/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_4/self_attention_layer/key/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_40/Initializer/ReadVariableOpReadVariableOp5transformer_layer_4/self_attention_layer/key/kernel_1*
_class
loc:@Variable_40*#
_output_shapes
:�@*
dtype0
�
Variable_40VarHandleOp*
_class
loc:@Variable_40*
_output_shapes
: *

debug_nameVariable_40/*
dtype0*
shape:�@*
shared_nameVariable_40
g
,Variable_40/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_40*
_output_shapes
: 
h
Variable_40/AssignAssignVariableOpVariable_40&Variable_40/Initializer/ReadVariableOp*
dtype0
p
Variable_40/Read/ReadVariableOpReadVariableOpVariable_40*#
_output_shapes
:�@*
dtype0
�
5transformer_layer_4/self_attention_layer/query/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_4/self_attention_layer/query/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_4/self_attention_layer/query/bias_1
�
Itransformer_layer_4/self_attention_layer/query/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_4/self_attention_layer/query/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_41/Initializer/ReadVariableOpReadVariableOp5transformer_layer_4/self_attention_layer/query/bias_1*
_class
loc:@Variable_41*
_output_shapes

:@*
dtype0
�
Variable_41VarHandleOp*
_class
loc:@Variable_41*
_output_shapes
: *

debug_nameVariable_41/*
dtype0*
shape
:@*
shared_nameVariable_41
g
,Variable_41/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_41*
_output_shapes
: 
h
Variable_41/AssignAssignVariableOpVariable_41&Variable_41/Initializer/ReadVariableOp*
dtype0
k
Variable_41/Read/ReadVariableOpReadVariableOpVariable_41*
_output_shapes

:@*
dtype0
�
7transformer_layer_4/self_attention_layer/query/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_4/self_attention_layer/query/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_4/self_attention_layer/query/kernel_1
�
Ktransformer_layer_4/self_attention_layer/query/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_4/self_attention_layer/query/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_42/Initializer/ReadVariableOpReadVariableOp7transformer_layer_4/self_attention_layer/query/kernel_1*
_class
loc:@Variable_42*#
_output_shapes
:�@*
dtype0
�
Variable_42VarHandleOp*
_class
loc:@Variable_42*
_output_shapes
: *

debug_nameVariable_42/*
dtype0*
shape:�@*
shared_nameVariable_42
g
,Variable_42/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_42*
_output_shapes
: 
h
Variable_42/AssignAssignVariableOpVariable_42&Variable_42/Initializer/ReadVariableOp*
dtype0
p
Variable_42/Read/ReadVariableOpReadVariableOpVariable_42*#
_output_shapes
:�@*
dtype0
�
(transformer_layer_3/seed_generator_stateVarHandleOp*
_output_shapes
: *9

debug_name+)transformer_layer_3/seed_generator_state/*
dtype0*
shape:*9
shared_name*(transformer_layer_3/seed_generator_state
�
<transformer_layer_3/seed_generator_state/Read/ReadVariableOpReadVariableOp(transformer_layer_3/seed_generator_state*
_output_shapes
:*
dtype0
�
&Variable_43/Initializer/ReadVariableOpReadVariableOp(transformer_layer_3/seed_generator_state*
_class
loc:@Variable_43*
_output_shapes
:*
dtype0
�
Variable_43VarHandleOp*
_class
loc:@Variable_43*
_output_shapes
: *

debug_nameVariable_43/*
dtype0*
shape:*
shared_nameVariable_43
g
,Variable_43/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_43*
_output_shapes
: 
h
Variable_43/AssignAssignVariableOpVariable_43&Variable_43/Initializer/ReadVariableOp*
dtype0
g
Variable_43/Read/ReadVariableOpReadVariableOpVariable_43*
_output_shapes
:*
dtype0
�
3transformer_layer_3/feedforward_output_dense/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_3/feedforward_output_dense/bias_1/*
dtype0*
shape:�*D
shared_name53transformer_layer_3/feedforward_output_dense/bias_1
�
Gtransformer_layer_3/feedforward_output_dense/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_3/feedforward_output_dense/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_44/Initializer/ReadVariableOpReadVariableOp3transformer_layer_3/feedforward_output_dense/bias_1*
_class
loc:@Variable_44*
_output_shapes	
:�*
dtype0
�
Variable_44VarHandleOp*
_class
loc:@Variable_44*
_output_shapes
: *

debug_nameVariable_44/*
dtype0*
shape:�*
shared_nameVariable_44
g
,Variable_44/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_44*
_output_shapes
: 
h
Variable_44/AssignAssignVariableOpVariable_44&Variable_44/Initializer/ReadVariableOp*
dtype0
h
Variable_44/Read/ReadVariableOpReadVariableOpVariable_44*
_output_shapes	
:�*
dtype0
�
5transformer_layer_3/feedforward_output_dense/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_3/feedforward_output_dense/kernel_1/*
dtype0*
shape:
��*F
shared_name75transformer_layer_3/feedforward_output_dense/kernel_1
�
Itransformer_layer_3/feedforward_output_dense/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_3/feedforward_output_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_45/Initializer/ReadVariableOpReadVariableOp5transformer_layer_3/feedforward_output_dense/kernel_1*
_class
loc:@Variable_45* 
_output_shapes
:
��*
dtype0
�
Variable_45VarHandleOp*
_class
loc:@Variable_45*
_output_shapes
: *

debug_nameVariable_45/*
dtype0*
shape:
��*
shared_nameVariable_45
g
,Variable_45/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_45*
_output_shapes
: 
h
Variable_45/AssignAssignVariableOpVariable_45&Variable_45/Initializer/ReadVariableOp*
dtype0
m
Variable_45/Read/ReadVariableOpReadVariableOpVariable_45* 
_output_shapes
:
��*
dtype0
�
9transformer_layer_3/feedforward_intermediate_dense/bias_1VarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_3/feedforward_intermediate_dense/bias_1/*
dtype0*
shape:�*J
shared_name;9transformer_layer_3/feedforward_intermediate_dense/bias_1
�
Mtransformer_layer_3/feedforward_intermediate_dense/bias_1/Read/ReadVariableOpReadVariableOp9transformer_layer_3/feedforward_intermediate_dense/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_46/Initializer/ReadVariableOpReadVariableOp9transformer_layer_3/feedforward_intermediate_dense/bias_1*
_class
loc:@Variable_46*
_output_shapes	
:�*
dtype0
�
Variable_46VarHandleOp*
_class
loc:@Variable_46*
_output_shapes
: *

debug_nameVariable_46/*
dtype0*
shape:�*
shared_nameVariable_46
g
,Variable_46/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_46*
_output_shapes
: 
h
Variable_46/AssignAssignVariableOpVariable_46&Variable_46/Initializer/ReadVariableOp*
dtype0
h
Variable_46/Read/ReadVariableOpReadVariableOpVariable_46*
_output_shapes	
:�*
dtype0
�
;transformer_layer_3/feedforward_intermediate_dense/kernel_1VarHandleOp*
_output_shapes
: *L

debug_name><transformer_layer_3/feedforward_intermediate_dense/kernel_1/*
dtype0*
shape:
��*L
shared_name=;transformer_layer_3/feedforward_intermediate_dense/kernel_1
�
Otransformer_layer_3/feedforward_intermediate_dense/kernel_1/Read/ReadVariableOpReadVariableOp;transformer_layer_3/feedforward_intermediate_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_47/Initializer/ReadVariableOpReadVariableOp;transformer_layer_3/feedforward_intermediate_dense/kernel_1*
_class
loc:@Variable_47* 
_output_shapes
:
��*
dtype0
�
Variable_47VarHandleOp*
_class
loc:@Variable_47*
_output_shapes
: *

debug_nameVariable_47/*
dtype0*
shape:
��*
shared_nameVariable_47
g
,Variable_47/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_47*
_output_shapes
: 
h
Variable_47/AssignAssignVariableOpVariable_47&Variable_47/Initializer/ReadVariableOp*
dtype0
m
Variable_47/Read/ReadVariableOpReadVariableOpVariable_47* 
_output_shapes
:
��*
dtype0
�
1transformer_layer_3/feedforward_layer_norm/beta_1VarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_3/feedforward_layer_norm/beta_1/*
dtype0*
shape:�*B
shared_name31transformer_layer_3/feedforward_layer_norm/beta_1
�
Etransformer_layer_3/feedforward_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp1transformer_layer_3/feedforward_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_48/Initializer/ReadVariableOpReadVariableOp1transformer_layer_3/feedforward_layer_norm/beta_1*
_class
loc:@Variable_48*
_output_shapes	
:�*
dtype0
�
Variable_48VarHandleOp*
_class
loc:@Variable_48*
_output_shapes
: *

debug_nameVariable_48/*
dtype0*
shape:�*
shared_nameVariable_48
g
,Variable_48/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_48*
_output_shapes
: 
h
Variable_48/AssignAssignVariableOpVariable_48&Variable_48/Initializer/ReadVariableOp*
dtype0
h
Variable_48/Read/ReadVariableOpReadVariableOpVariable_48*
_output_shapes	
:�*
dtype0
�
2transformer_layer_3/feedforward_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_3/feedforward_layer_norm/gamma_1/*
dtype0*
shape:�*C
shared_name42transformer_layer_3/feedforward_layer_norm/gamma_1
�
Ftransformer_layer_3/feedforward_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp2transformer_layer_3/feedforward_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_49/Initializer/ReadVariableOpReadVariableOp2transformer_layer_3/feedforward_layer_norm/gamma_1*
_class
loc:@Variable_49*
_output_shapes	
:�*
dtype0
�
Variable_49VarHandleOp*
_class
loc:@Variable_49*
_output_shapes
: *

debug_nameVariable_49/*
dtype0*
shape:�*
shared_nameVariable_49
g
,Variable_49/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_49*
_output_shapes
: 
h
Variable_49/AssignAssignVariableOpVariable_49&Variable_49/Initializer/ReadVariableOp*
dtype0
h
Variable_49/Read/ReadVariableOpReadVariableOpVariable_49*
_output_shapes	
:�*
dtype0
�
*transformer_layer_3/seed_generator_state_1VarHandleOp*
_output_shapes
: *;

debug_name-+transformer_layer_3/seed_generator_state_1/*
dtype0*
shape:*;
shared_name,*transformer_layer_3/seed_generator_state_1
�
>transformer_layer_3/seed_generator_state_1/Read/ReadVariableOpReadVariableOp*transformer_layer_3/seed_generator_state_1*
_output_shapes
:*
dtype0
�
&Variable_50/Initializer/ReadVariableOpReadVariableOp*transformer_layer_3/seed_generator_state_1*
_class
loc:@Variable_50*
_output_shapes
:*
dtype0
�
Variable_50VarHandleOp*
_class
loc:@Variable_50*
_output_shapes
: *

debug_nameVariable_50/*
dtype0*
shape:*
shared_nameVariable_50
g
,Variable_50/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_50*
_output_shapes
: 
h
Variable_50/AssignAssignVariableOpVariable_50&Variable_50/Initializer/ReadVariableOp*
dtype0
g
Variable_50/Read/ReadVariableOpReadVariableOpVariable_50*
_output_shapes
:*
dtype0
�
4transformer_layer_3/self_attention_layer_norm/beta_1VarHandleOp*
_output_shapes
: *E

debug_name75transformer_layer_3/self_attention_layer_norm/beta_1/*
dtype0*
shape:�*E
shared_name64transformer_layer_3/self_attention_layer_norm/beta_1
�
Htransformer_layer_3/self_attention_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp4transformer_layer_3/self_attention_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_51/Initializer/ReadVariableOpReadVariableOp4transformer_layer_3/self_attention_layer_norm/beta_1*
_class
loc:@Variable_51*
_output_shapes	
:�*
dtype0
�
Variable_51VarHandleOp*
_class
loc:@Variable_51*
_output_shapes
: *

debug_nameVariable_51/*
dtype0*
shape:�*
shared_nameVariable_51
g
,Variable_51/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_51*
_output_shapes
: 
h
Variable_51/AssignAssignVariableOpVariable_51&Variable_51/Initializer/ReadVariableOp*
dtype0
h
Variable_51/Read/ReadVariableOpReadVariableOpVariable_51*
_output_shapes	
:�*
dtype0
�
5transformer_layer_3/self_attention_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_3/self_attention_layer_norm/gamma_1/*
dtype0*
shape:�*F
shared_name75transformer_layer_3/self_attention_layer_norm/gamma_1
�
Itransformer_layer_3/self_attention_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp5transformer_layer_3/self_attention_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_52/Initializer/ReadVariableOpReadVariableOp5transformer_layer_3/self_attention_layer_norm/gamma_1*
_class
loc:@Variable_52*
_output_shapes	
:�*
dtype0
�
Variable_52VarHandleOp*
_class
loc:@Variable_52*
_output_shapes
: *

debug_nameVariable_52/*
dtype0*
shape:�*
shared_nameVariable_52
g
,Variable_52/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_52*
_output_shapes
: 
h
Variable_52/AssignAssignVariableOpVariable_52&Variable_52/Initializer/ReadVariableOp*
dtype0
h
Variable_52/Read/ReadVariableOpReadVariableOpVariable_52*
_output_shapes	
:�*
dtype0
�
@transformer_layer_3/self_attention_layer/attention_output/bias_1VarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_3/self_attention_layer/attention_output/bias_1/*
dtype0*
shape:�*Q
shared_nameB@transformer_layer_3/self_attention_layer/attention_output/bias_1
�
Ttransformer_layer_3/self_attention_layer/attention_output/bias_1/Read/ReadVariableOpReadVariableOp@transformer_layer_3/self_attention_layer/attention_output/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_53/Initializer/ReadVariableOpReadVariableOp@transformer_layer_3/self_attention_layer/attention_output/bias_1*
_class
loc:@Variable_53*
_output_shapes	
:�*
dtype0
�
Variable_53VarHandleOp*
_class
loc:@Variable_53*
_output_shapes
: *

debug_nameVariable_53/*
dtype0*
shape:�*
shared_nameVariable_53
g
,Variable_53/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_53*
_output_shapes
: 
h
Variable_53/AssignAssignVariableOpVariable_53&Variable_53/Initializer/ReadVariableOp*
dtype0
h
Variable_53/Read/ReadVariableOpReadVariableOpVariable_53*
_output_shapes	
:�*
dtype0
�
Btransformer_layer_3/self_attention_layer/attention_output/kernel_1VarHandleOp*
_output_shapes
: *S

debug_nameECtransformer_layer_3/self_attention_layer/attention_output/kernel_1/*
dtype0*
shape:@�*S
shared_nameDBtransformer_layer_3/self_attention_layer/attention_output/kernel_1
�
Vtransformer_layer_3/self_attention_layer/attention_output/kernel_1/Read/ReadVariableOpReadVariableOpBtransformer_layer_3/self_attention_layer/attention_output/kernel_1*#
_output_shapes
:@�*
dtype0
�
&Variable_54/Initializer/ReadVariableOpReadVariableOpBtransformer_layer_3/self_attention_layer/attention_output/kernel_1*
_class
loc:@Variable_54*#
_output_shapes
:@�*
dtype0
�
Variable_54VarHandleOp*
_class
loc:@Variable_54*
_output_shapes
: *

debug_nameVariable_54/*
dtype0*
shape:@�*
shared_nameVariable_54
g
,Variable_54/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_54*
_output_shapes
: 
h
Variable_54/AssignAssignVariableOpVariable_54&Variable_54/Initializer/ReadVariableOp*
dtype0
p
Variable_54/Read/ReadVariableOpReadVariableOpVariable_54*#
_output_shapes
:@�*
dtype0
�
=transformer_layer_3/self_attention_layer/seed_generator_stateVarHandleOp*
_output_shapes
: *N

debug_name@>transformer_layer_3/self_attention_layer/seed_generator_state/*
dtype0*
shape:*N
shared_name?=transformer_layer_3/self_attention_layer/seed_generator_state
�
Qtransformer_layer_3/self_attention_layer/seed_generator_state/Read/ReadVariableOpReadVariableOp=transformer_layer_3/self_attention_layer/seed_generator_state*
_output_shapes
:*
dtype0
�
&Variable_55/Initializer/ReadVariableOpReadVariableOp=transformer_layer_3/self_attention_layer/seed_generator_state*
_class
loc:@Variable_55*
_output_shapes
:*
dtype0
�
Variable_55VarHandleOp*
_class
loc:@Variable_55*
_output_shapes
: *

debug_nameVariable_55/*
dtype0*
shape:*
shared_nameVariable_55
g
,Variable_55/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_55*
_output_shapes
: 
h
Variable_55/AssignAssignVariableOpVariable_55&Variable_55/Initializer/ReadVariableOp*
dtype0
g
Variable_55/Read/ReadVariableOpReadVariableOpVariable_55*
_output_shapes
:*
dtype0
�
5transformer_layer_3/self_attention_layer/value/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_3/self_attention_layer/value/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_3/self_attention_layer/value/bias_1
�
Itransformer_layer_3/self_attention_layer/value/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_3/self_attention_layer/value/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_56/Initializer/ReadVariableOpReadVariableOp5transformer_layer_3/self_attention_layer/value/bias_1*
_class
loc:@Variable_56*
_output_shapes

:@*
dtype0
�
Variable_56VarHandleOp*
_class
loc:@Variable_56*
_output_shapes
: *

debug_nameVariable_56/*
dtype0*
shape
:@*
shared_nameVariable_56
g
,Variable_56/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_56*
_output_shapes
: 
h
Variable_56/AssignAssignVariableOpVariable_56&Variable_56/Initializer/ReadVariableOp*
dtype0
k
Variable_56/Read/ReadVariableOpReadVariableOpVariable_56*
_output_shapes

:@*
dtype0
�
7transformer_layer_3/self_attention_layer/value/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_3/self_attention_layer/value/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_3/self_attention_layer/value/kernel_1
�
Ktransformer_layer_3/self_attention_layer/value/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_3/self_attention_layer/value/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_57/Initializer/ReadVariableOpReadVariableOp7transformer_layer_3/self_attention_layer/value/kernel_1*
_class
loc:@Variable_57*#
_output_shapes
:�@*
dtype0
�
Variable_57VarHandleOp*
_class
loc:@Variable_57*
_output_shapes
: *

debug_nameVariable_57/*
dtype0*
shape:�@*
shared_nameVariable_57
g
,Variable_57/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_57*
_output_shapes
: 
h
Variable_57/AssignAssignVariableOpVariable_57&Variable_57/Initializer/ReadVariableOp*
dtype0
p
Variable_57/Read/ReadVariableOpReadVariableOpVariable_57*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_3/self_attention_layer/key/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_3/self_attention_layer/key/bias_1/*
dtype0*
shape
:@*D
shared_name53transformer_layer_3/self_attention_layer/key/bias_1
�
Gtransformer_layer_3/self_attention_layer/key/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_3/self_attention_layer/key/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_58/Initializer/ReadVariableOpReadVariableOp3transformer_layer_3/self_attention_layer/key/bias_1*
_class
loc:@Variable_58*
_output_shapes

:@*
dtype0
�
Variable_58VarHandleOp*
_class
loc:@Variable_58*
_output_shapes
: *

debug_nameVariable_58/*
dtype0*
shape
:@*
shared_nameVariable_58
g
,Variable_58/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_58*
_output_shapes
: 
h
Variable_58/AssignAssignVariableOpVariable_58&Variable_58/Initializer/ReadVariableOp*
dtype0
k
Variable_58/Read/ReadVariableOpReadVariableOpVariable_58*
_output_shapes

:@*
dtype0
�
5transformer_layer_3/self_attention_layer/key/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_3/self_attention_layer/key/kernel_1/*
dtype0*
shape:�@*F
shared_name75transformer_layer_3/self_attention_layer/key/kernel_1
�
Itransformer_layer_3/self_attention_layer/key/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_3/self_attention_layer/key/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_59/Initializer/ReadVariableOpReadVariableOp5transformer_layer_3/self_attention_layer/key/kernel_1*
_class
loc:@Variable_59*#
_output_shapes
:�@*
dtype0
�
Variable_59VarHandleOp*
_class
loc:@Variable_59*
_output_shapes
: *

debug_nameVariable_59/*
dtype0*
shape:�@*
shared_nameVariable_59
g
,Variable_59/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_59*
_output_shapes
: 
h
Variable_59/AssignAssignVariableOpVariable_59&Variable_59/Initializer/ReadVariableOp*
dtype0
p
Variable_59/Read/ReadVariableOpReadVariableOpVariable_59*#
_output_shapes
:�@*
dtype0
�
5transformer_layer_3/self_attention_layer/query/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_3/self_attention_layer/query/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_3/self_attention_layer/query/bias_1
�
Itransformer_layer_3/self_attention_layer/query/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_3/self_attention_layer/query/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_60/Initializer/ReadVariableOpReadVariableOp5transformer_layer_3/self_attention_layer/query/bias_1*
_class
loc:@Variable_60*
_output_shapes

:@*
dtype0
�
Variable_60VarHandleOp*
_class
loc:@Variable_60*
_output_shapes
: *

debug_nameVariable_60/*
dtype0*
shape
:@*
shared_nameVariable_60
g
,Variable_60/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_60*
_output_shapes
: 
h
Variable_60/AssignAssignVariableOpVariable_60&Variable_60/Initializer/ReadVariableOp*
dtype0
k
Variable_60/Read/ReadVariableOpReadVariableOpVariable_60*
_output_shapes

:@*
dtype0
�
7transformer_layer_3/self_attention_layer/query/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_3/self_attention_layer/query/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_3/self_attention_layer/query/kernel_1
�
Ktransformer_layer_3/self_attention_layer/query/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_3/self_attention_layer/query/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_61/Initializer/ReadVariableOpReadVariableOp7transformer_layer_3/self_attention_layer/query/kernel_1*
_class
loc:@Variable_61*#
_output_shapes
:�@*
dtype0
�
Variable_61VarHandleOp*
_class
loc:@Variable_61*
_output_shapes
: *

debug_nameVariable_61/*
dtype0*
shape:�@*
shared_nameVariable_61
g
,Variable_61/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_61*
_output_shapes
: 
h
Variable_61/AssignAssignVariableOpVariable_61&Variable_61/Initializer/ReadVariableOp*
dtype0
p
Variable_61/Read/ReadVariableOpReadVariableOpVariable_61*#
_output_shapes
:�@*
dtype0
�
(transformer_layer_2/seed_generator_stateVarHandleOp*
_output_shapes
: *9

debug_name+)transformer_layer_2/seed_generator_state/*
dtype0*
shape:*9
shared_name*(transformer_layer_2/seed_generator_state
�
<transformer_layer_2/seed_generator_state/Read/ReadVariableOpReadVariableOp(transformer_layer_2/seed_generator_state*
_output_shapes
:*
dtype0
�
&Variable_62/Initializer/ReadVariableOpReadVariableOp(transformer_layer_2/seed_generator_state*
_class
loc:@Variable_62*
_output_shapes
:*
dtype0
�
Variable_62VarHandleOp*
_class
loc:@Variable_62*
_output_shapes
: *

debug_nameVariable_62/*
dtype0*
shape:*
shared_nameVariable_62
g
,Variable_62/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_62*
_output_shapes
: 
h
Variable_62/AssignAssignVariableOpVariable_62&Variable_62/Initializer/ReadVariableOp*
dtype0
g
Variable_62/Read/ReadVariableOpReadVariableOpVariable_62*
_output_shapes
:*
dtype0
�
3transformer_layer_2/feedforward_output_dense/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_2/feedforward_output_dense/bias_1/*
dtype0*
shape:�*D
shared_name53transformer_layer_2/feedforward_output_dense/bias_1
�
Gtransformer_layer_2/feedforward_output_dense/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_2/feedforward_output_dense/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_63/Initializer/ReadVariableOpReadVariableOp3transformer_layer_2/feedforward_output_dense/bias_1*
_class
loc:@Variable_63*
_output_shapes	
:�*
dtype0
�
Variable_63VarHandleOp*
_class
loc:@Variable_63*
_output_shapes
: *

debug_nameVariable_63/*
dtype0*
shape:�*
shared_nameVariable_63
g
,Variable_63/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_63*
_output_shapes
: 
h
Variable_63/AssignAssignVariableOpVariable_63&Variable_63/Initializer/ReadVariableOp*
dtype0
h
Variable_63/Read/ReadVariableOpReadVariableOpVariable_63*
_output_shapes	
:�*
dtype0
�
5transformer_layer_2/feedforward_output_dense/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_2/feedforward_output_dense/kernel_1/*
dtype0*
shape:
��*F
shared_name75transformer_layer_2/feedforward_output_dense/kernel_1
�
Itransformer_layer_2/feedforward_output_dense/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_2/feedforward_output_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_64/Initializer/ReadVariableOpReadVariableOp5transformer_layer_2/feedforward_output_dense/kernel_1*
_class
loc:@Variable_64* 
_output_shapes
:
��*
dtype0
�
Variable_64VarHandleOp*
_class
loc:@Variable_64*
_output_shapes
: *

debug_nameVariable_64/*
dtype0*
shape:
��*
shared_nameVariable_64
g
,Variable_64/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_64*
_output_shapes
: 
h
Variable_64/AssignAssignVariableOpVariable_64&Variable_64/Initializer/ReadVariableOp*
dtype0
m
Variable_64/Read/ReadVariableOpReadVariableOpVariable_64* 
_output_shapes
:
��*
dtype0
�
9transformer_layer_2/feedforward_intermediate_dense/bias_1VarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_2/feedforward_intermediate_dense/bias_1/*
dtype0*
shape:�*J
shared_name;9transformer_layer_2/feedforward_intermediate_dense/bias_1
�
Mtransformer_layer_2/feedforward_intermediate_dense/bias_1/Read/ReadVariableOpReadVariableOp9transformer_layer_2/feedforward_intermediate_dense/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_65/Initializer/ReadVariableOpReadVariableOp9transformer_layer_2/feedforward_intermediate_dense/bias_1*
_class
loc:@Variable_65*
_output_shapes	
:�*
dtype0
�
Variable_65VarHandleOp*
_class
loc:@Variable_65*
_output_shapes
: *

debug_nameVariable_65/*
dtype0*
shape:�*
shared_nameVariable_65
g
,Variable_65/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_65*
_output_shapes
: 
h
Variable_65/AssignAssignVariableOpVariable_65&Variable_65/Initializer/ReadVariableOp*
dtype0
h
Variable_65/Read/ReadVariableOpReadVariableOpVariable_65*
_output_shapes	
:�*
dtype0
�
;transformer_layer_2/feedforward_intermediate_dense/kernel_1VarHandleOp*
_output_shapes
: *L

debug_name><transformer_layer_2/feedforward_intermediate_dense/kernel_1/*
dtype0*
shape:
��*L
shared_name=;transformer_layer_2/feedforward_intermediate_dense/kernel_1
�
Otransformer_layer_2/feedforward_intermediate_dense/kernel_1/Read/ReadVariableOpReadVariableOp;transformer_layer_2/feedforward_intermediate_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_66/Initializer/ReadVariableOpReadVariableOp;transformer_layer_2/feedforward_intermediate_dense/kernel_1*
_class
loc:@Variable_66* 
_output_shapes
:
��*
dtype0
�
Variable_66VarHandleOp*
_class
loc:@Variable_66*
_output_shapes
: *

debug_nameVariable_66/*
dtype0*
shape:
��*
shared_nameVariable_66
g
,Variable_66/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_66*
_output_shapes
: 
h
Variable_66/AssignAssignVariableOpVariable_66&Variable_66/Initializer/ReadVariableOp*
dtype0
m
Variable_66/Read/ReadVariableOpReadVariableOpVariable_66* 
_output_shapes
:
��*
dtype0
�
1transformer_layer_2/feedforward_layer_norm/beta_1VarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_2/feedforward_layer_norm/beta_1/*
dtype0*
shape:�*B
shared_name31transformer_layer_2/feedforward_layer_norm/beta_1
�
Etransformer_layer_2/feedforward_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp1transformer_layer_2/feedforward_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_67/Initializer/ReadVariableOpReadVariableOp1transformer_layer_2/feedforward_layer_norm/beta_1*
_class
loc:@Variable_67*
_output_shapes	
:�*
dtype0
�
Variable_67VarHandleOp*
_class
loc:@Variable_67*
_output_shapes
: *

debug_nameVariable_67/*
dtype0*
shape:�*
shared_nameVariable_67
g
,Variable_67/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_67*
_output_shapes
: 
h
Variable_67/AssignAssignVariableOpVariable_67&Variable_67/Initializer/ReadVariableOp*
dtype0
h
Variable_67/Read/ReadVariableOpReadVariableOpVariable_67*
_output_shapes	
:�*
dtype0
�
2transformer_layer_2/feedforward_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_2/feedforward_layer_norm/gamma_1/*
dtype0*
shape:�*C
shared_name42transformer_layer_2/feedforward_layer_norm/gamma_1
�
Ftransformer_layer_2/feedforward_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp2transformer_layer_2/feedforward_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_68/Initializer/ReadVariableOpReadVariableOp2transformer_layer_2/feedforward_layer_norm/gamma_1*
_class
loc:@Variable_68*
_output_shapes	
:�*
dtype0
�
Variable_68VarHandleOp*
_class
loc:@Variable_68*
_output_shapes
: *

debug_nameVariable_68/*
dtype0*
shape:�*
shared_nameVariable_68
g
,Variable_68/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_68*
_output_shapes
: 
h
Variable_68/AssignAssignVariableOpVariable_68&Variable_68/Initializer/ReadVariableOp*
dtype0
h
Variable_68/Read/ReadVariableOpReadVariableOpVariable_68*
_output_shapes	
:�*
dtype0
�
*transformer_layer_2/seed_generator_state_1VarHandleOp*
_output_shapes
: *;

debug_name-+transformer_layer_2/seed_generator_state_1/*
dtype0*
shape:*;
shared_name,*transformer_layer_2/seed_generator_state_1
�
>transformer_layer_2/seed_generator_state_1/Read/ReadVariableOpReadVariableOp*transformer_layer_2/seed_generator_state_1*
_output_shapes
:*
dtype0
�
&Variable_69/Initializer/ReadVariableOpReadVariableOp*transformer_layer_2/seed_generator_state_1*
_class
loc:@Variable_69*
_output_shapes
:*
dtype0
�
Variable_69VarHandleOp*
_class
loc:@Variable_69*
_output_shapes
: *

debug_nameVariable_69/*
dtype0*
shape:*
shared_nameVariable_69
g
,Variable_69/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_69*
_output_shapes
: 
h
Variable_69/AssignAssignVariableOpVariable_69&Variable_69/Initializer/ReadVariableOp*
dtype0
g
Variable_69/Read/ReadVariableOpReadVariableOpVariable_69*
_output_shapes
:*
dtype0
�
4transformer_layer_2/self_attention_layer_norm/beta_1VarHandleOp*
_output_shapes
: *E

debug_name75transformer_layer_2/self_attention_layer_norm/beta_1/*
dtype0*
shape:�*E
shared_name64transformer_layer_2/self_attention_layer_norm/beta_1
�
Htransformer_layer_2/self_attention_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp4transformer_layer_2/self_attention_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_70/Initializer/ReadVariableOpReadVariableOp4transformer_layer_2/self_attention_layer_norm/beta_1*
_class
loc:@Variable_70*
_output_shapes	
:�*
dtype0
�
Variable_70VarHandleOp*
_class
loc:@Variable_70*
_output_shapes
: *

debug_nameVariable_70/*
dtype0*
shape:�*
shared_nameVariable_70
g
,Variable_70/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_70*
_output_shapes
: 
h
Variable_70/AssignAssignVariableOpVariable_70&Variable_70/Initializer/ReadVariableOp*
dtype0
h
Variable_70/Read/ReadVariableOpReadVariableOpVariable_70*
_output_shapes	
:�*
dtype0
�
5transformer_layer_2/self_attention_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_2/self_attention_layer_norm/gamma_1/*
dtype0*
shape:�*F
shared_name75transformer_layer_2/self_attention_layer_norm/gamma_1
�
Itransformer_layer_2/self_attention_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp5transformer_layer_2/self_attention_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_71/Initializer/ReadVariableOpReadVariableOp5transformer_layer_2/self_attention_layer_norm/gamma_1*
_class
loc:@Variable_71*
_output_shapes	
:�*
dtype0
�
Variable_71VarHandleOp*
_class
loc:@Variable_71*
_output_shapes
: *

debug_nameVariable_71/*
dtype0*
shape:�*
shared_nameVariable_71
g
,Variable_71/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_71*
_output_shapes
: 
h
Variable_71/AssignAssignVariableOpVariable_71&Variable_71/Initializer/ReadVariableOp*
dtype0
h
Variable_71/Read/ReadVariableOpReadVariableOpVariable_71*
_output_shapes	
:�*
dtype0
�
@transformer_layer_2/self_attention_layer/attention_output/bias_1VarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_2/self_attention_layer/attention_output/bias_1/*
dtype0*
shape:�*Q
shared_nameB@transformer_layer_2/self_attention_layer/attention_output/bias_1
�
Ttransformer_layer_2/self_attention_layer/attention_output/bias_1/Read/ReadVariableOpReadVariableOp@transformer_layer_2/self_attention_layer/attention_output/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_72/Initializer/ReadVariableOpReadVariableOp@transformer_layer_2/self_attention_layer/attention_output/bias_1*
_class
loc:@Variable_72*
_output_shapes	
:�*
dtype0
�
Variable_72VarHandleOp*
_class
loc:@Variable_72*
_output_shapes
: *

debug_nameVariable_72/*
dtype0*
shape:�*
shared_nameVariable_72
g
,Variable_72/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_72*
_output_shapes
: 
h
Variable_72/AssignAssignVariableOpVariable_72&Variable_72/Initializer/ReadVariableOp*
dtype0
h
Variable_72/Read/ReadVariableOpReadVariableOpVariable_72*
_output_shapes	
:�*
dtype0
�
Btransformer_layer_2/self_attention_layer/attention_output/kernel_1VarHandleOp*
_output_shapes
: *S

debug_nameECtransformer_layer_2/self_attention_layer/attention_output/kernel_1/*
dtype0*
shape:@�*S
shared_nameDBtransformer_layer_2/self_attention_layer/attention_output/kernel_1
�
Vtransformer_layer_2/self_attention_layer/attention_output/kernel_1/Read/ReadVariableOpReadVariableOpBtransformer_layer_2/self_attention_layer/attention_output/kernel_1*#
_output_shapes
:@�*
dtype0
�
&Variable_73/Initializer/ReadVariableOpReadVariableOpBtransformer_layer_2/self_attention_layer/attention_output/kernel_1*
_class
loc:@Variable_73*#
_output_shapes
:@�*
dtype0
�
Variable_73VarHandleOp*
_class
loc:@Variable_73*
_output_shapes
: *

debug_nameVariable_73/*
dtype0*
shape:@�*
shared_nameVariable_73
g
,Variable_73/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_73*
_output_shapes
: 
h
Variable_73/AssignAssignVariableOpVariable_73&Variable_73/Initializer/ReadVariableOp*
dtype0
p
Variable_73/Read/ReadVariableOpReadVariableOpVariable_73*#
_output_shapes
:@�*
dtype0
�
=transformer_layer_2/self_attention_layer/seed_generator_stateVarHandleOp*
_output_shapes
: *N

debug_name@>transformer_layer_2/self_attention_layer/seed_generator_state/*
dtype0*
shape:*N
shared_name?=transformer_layer_2/self_attention_layer/seed_generator_state
�
Qtransformer_layer_2/self_attention_layer/seed_generator_state/Read/ReadVariableOpReadVariableOp=transformer_layer_2/self_attention_layer/seed_generator_state*
_output_shapes
:*
dtype0
�
&Variable_74/Initializer/ReadVariableOpReadVariableOp=transformer_layer_2/self_attention_layer/seed_generator_state*
_class
loc:@Variable_74*
_output_shapes
:*
dtype0
�
Variable_74VarHandleOp*
_class
loc:@Variable_74*
_output_shapes
: *

debug_nameVariable_74/*
dtype0*
shape:*
shared_nameVariable_74
g
,Variable_74/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_74*
_output_shapes
: 
h
Variable_74/AssignAssignVariableOpVariable_74&Variable_74/Initializer/ReadVariableOp*
dtype0
g
Variable_74/Read/ReadVariableOpReadVariableOpVariable_74*
_output_shapes
:*
dtype0
�
5transformer_layer_2/self_attention_layer/value/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_2/self_attention_layer/value/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_2/self_attention_layer/value/bias_1
�
Itransformer_layer_2/self_attention_layer/value/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_2/self_attention_layer/value/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_75/Initializer/ReadVariableOpReadVariableOp5transformer_layer_2/self_attention_layer/value/bias_1*
_class
loc:@Variable_75*
_output_shapes

:@*
dtype0
�
Variable_75VarHandleOp*
_class
loc:@Variable_75*
_output_shapes
: *

debug_nameVariable_75/*
dtype0*
shape
:@*
shared_nameVariable_75
g
,Variable_75/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_75*
_output_shapes
: 
h
Variable_75/AssignAssignVariableOpVariable_75&Variable_75/Initializer/ReadVariableOp*
dtype0
k
Variable_75/Read/ReadVariableOpReadVariableOpVariable_75*
_output_shapes

:@*
dtype0
�
7transformer_layer_2/self_attention_layer/value/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_2/self_attention_layer/value/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_2/self_attention_layer/value/kernel_1
�
Ktransformer_layer_2/self_attention_layer/value/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_2/self_attention_layer/value/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_76/Initializer/ReadVariableOpReadVariableOp7transformer_layer_2/self_attention_layer/value/kernel_1*
_class
loc:@Variable_76*#
_output_shapes
:�@*
dtype0
�
Variable_76VarHandleOp*
_class
loc:@Variable_76*
_output_shapes
: *

debug_nameVariable_76/*
dtype0*
shape:�@*
shared_nameVariable_76
g
,Variable_76/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_76*
_output_shapes
: 
h
Variable_76/AssignAssignVariableOpVariable_76&Variable_76/Initializer/ReadVariableOp*
dtype0
p
Variable_76/Read/ReadVariableOpReadVariableOpVariable_76*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_2/self_attention_layer/key/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_2/self_attention_layer/key/bias_1/*
dtype0*
shape
:@*D
shared_name53transformer_layer_2/self_attention_layer/key/bias_1
�
Gtransformer_layer_2/self_attention_layer/key/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_2/self_attention_layer/key/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_77/Initializer/ReadVariableOpReadVariableOp3transformer_layer_2/self_attention_layer/key/bias_1*
_class
loc:@Variable_77*
_output_shapes

:@*
dtype0
�
Variable_77VarHandleOp*
_class
loc:@Variable_77*
_output_shapes
: *

debug_nameVariable_77/*
dtype0*
shape
:@*
shared_nameVariable_77
g
,Variable_77/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_77*
_output_shapes
: 
h
Variable_77/AssignAssignVariableOpVariable_77&Variable_77/Initializer/ReadVariableOp*
dtype0
k
Variable_77/Read/ReadVariableOpReadVariableOpVariable_77*
_output_shapes

:@*
dtype0
�
5transformer_layer_2/self_attention_layer/key/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_2/self_attention_layer/key/kernel_1/*
dtype0*
shape:�@*F
shared_name75transformer_layer_2/self_attention_layer/key/kernel_1
�
Itransformer_layer_2/self_attention_layer/key/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_2/self_attention_layer/key/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_78/Initializer/ReadVariableOpReadVariableOp5transformer_layer_2/self_attention_layer/key/kernel_1*
_class
loc:@Variable_78*#
_output_shapes
:�@*
dtype0
�
Variable_78VarHandleOp*
_class
loc:@Variable_78*
_output_shapes
: *

debug_nameVariable_78/*
dtype0*
shape:�@*
shared_nameVariable_78
g
,Variable_78/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_78*
_output_shapes
: 
h
Variable_78/AssignAssignVariableOpVariable_78&Variable_78/Initializer/ReadVariableOp*
dtype0
p
Variable_78/Read/ReadVariableOpReadVariableOpVariable_78*#
_output_shapes
:�@*
dtype0
�
5transformer_layer_2/self_attention_layer/query/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_2/self_attention_layer/query/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_2/self_attention_layer/query/bias_1
�
Itransformer_layer_2/self_attention_layer/query/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_2/self_attention_layer/query/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_79/Initializer/ReadVariableOpReadVariableOp5transformer_layer_2/self_attention_layer/query/bias_1*
_class
loc:@Variable_79*
_output_shapes

:@*
dtype0
�
Variable_79VarHandleOp*
_class
loc:@Variable_79*
_output_shapes
: *

debug_nameVariable_79/*
dtype0*
shape
:@*
shared_nameVariable_79
g
,Variable_79/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_79*
_output_shapes
: 
h
Variable_79/AssignAssignVariableOpVariable_79&Variable_79/Initializer/ReadVariableOp*
dtype0
k
Variable_79/Read/ReadVariableOpReadVariableOpVariable_79*
_output_shapes

:@*
dtype0
�
7transformer_layer_2/self_attention_layer/query/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_2/self_attention_layer/query/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_2/self_attention_layer/query/kernel_1
�
Ktransformer_layer_2/self_attention_layer/query/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_2/self_attention_layer/query/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_80/Initializer/ReadVariableOpReadVariableOp7transformer_layer_2/self_attention_layer/query/kernel_1*
_class
loc:@Variable_80*#
_output_shapes
:�@*
dtype0
�
Variable_80VarHandleOp*
_class
loc:@Variable_80*
_output_shapes
: *

debug_nameVariable_80/*
dtype0*
shape:�@*
shared_nameVariable_80
g
,Variable_80/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_80*
_output_shapes
: 
h
Variable_80/AssignAssignVariableOpVariable_80&Variable_80/Initializer/ReadVariableOp*
dtype0
p
Variable_80/Read/ReadVariableOpReadVariableOpVariable_80*#
_output_shapes
:�@*
dtype0
�
(transformer_layer_1/seed_generator_stateVarHandleOp*
_output_shapes
: *9

debug_name+)transformer_layer_1/seed_generator_state/*
dtype0*
shape:*9
shared_name*(transformer_layer_1/seed_generator_state
�
<transformer_layer_1/seed_generator_state/Read/ReadVariableOpReadVariableOp(transformer_layer_1/seed_generator_state*
_output_shapes
:*
dtype0
�
&Variable_81/Initializer/ReadVariableOpReadVariableOp(transformer_layer_1/seed_generator_state*
_class
loc:@Variable_81*
_output_shapes
:*
dtype0
�
Variable_81VarHandleOp*
_class
loc:@Variable_81*
_output_shapes
: *

debug_nameVariable_81/*
dtype0*
shape:*
shared_nameVariable_81
g
,Variable_81/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_81*
_output_shapes
: 
h
Variable_81/AssignAssignVariableOpVariable_81&Variable_81/Initializer/ReadVariableOp*
dtype0
g
Variable_81/Read/ReadVariableOpReadVariableOpVariable_81*
_output_shapes
:*
dtype0
�
3transformer_layer_1/feedforward_output_dense/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_1/feedforward_output_dense/bias_1/*
dtype0*
shape:�*D
shared_name53transformer_layer_1/feedforward_output_dense/bias_1
�
Gtransformer_layer_1/feedforward_output_dense/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_1/feedforward_output_dense/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_82/Initializer/ReadVariableOpReadVariableOp3transformer_layer_1/feedforward_output_dense/bias_1*
_class
loc:@Variable_82*
_output_shapes	
:�*
dtype0
�
Variable_82VarHandleOp*
_class
loc:@Variable_82*
_output_shapes
: *

debug_nameVariable_82/*
dtype0*
shape:�*
shared_nameVariable_82
g
,Variable_82/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_82*
_output_shapes
: 
h
Variable_82/AssignAssignVariableOpVariable_82&Variable_82/Initializer/ReadVariableOp*
dtype0
h
Variable_82/Read/ReadVariableOpReadVariableOpVariable_82*
_output_shapes	
:�*
dtype0
�
5transformer_layer_1/feedforward_output_dense/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_1/feedforward_output_dense/kernel_1/*
dtype0*
shape:
��*F
shared_name75transformer_layer_1/feedforward_output_dense/kernel_1
�
Itransformer_layer_1/feedforward_output_dense/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_1/feedforward_output_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_83/Initializer/ReadVariableOpReadVariableOp5transformer_layer_1/feedforward_output_dense/kernel_1*
_class
loc:@Variable_83* 
_output_shapes
:
��*
dtype0
�
Variable_83VarHandleOp*
_class
loc:@Variable_83*
_output_shapes
: *

debug_nameVariable_83/*
dtype0*
shape:
��*
shared_nameVariable_83
g
,Variable_83/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_83*
_output_shapes
: 
h
Variable_83/AssignAssignVariableOpVariable_83&Variable_83/Initializer/ReadVariableOp*
dtype0
m
Variable_83/Read/ReadVariableOpReadVariableOpVariable_83* 
_output_shapes
:
��*
dtype0
�
9transformer_layer_1/feedforward_intermediate_dense/bias_1VarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_1/feedforward_intermediate_dense/bias_1/*
dtype0*
shape:�*J
shared_name;9transformer_layer_1/feedforward_intermediate_dense/bias_1
�
Mtransformer_layer_1/feedforward_intermediate_dense/bias_1/Read/ReadVariableOpReadVariableOp9transformer_layer_1/feedforward_intermediate_dense/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_84/Initializer/ReadVariableOpReadVariableOp9transformer_layer_1/feedforward_intermediate_dense/bias_1*
_class
loc:@Variable_84*
_output_shapes	
:�*
dtype0
�
Variable_84VarHandleOp*
_class
loc:@Variable_84*
_output_shapes
: *

debug_nameVariable_84/*
dtype0*
shape:�*
shared_nameVariable_84
g
,Variable_84/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_84*
_output_shapes
: 
h
Variable_84/AssignAssignVariableOpVariable_84&Variable_84/Initializer/ReadVariableOp*
dtype0
h
Variable_84/Read/ReadVariableOpReadVariableOpVariable_84*
_output_shapes	
:�*
dtype0
�
;transformer_layer_1/feedforward_intermediate_dense/kernel_1VarHandleOp*
_output_shapes
: *L

debug_name><transformer_layer_1/feedforward_intermediate_dense/kernel_1/*
dtype0*
shape:
��*L
shared_name=;transformer_layer_1/feedforward_intermediate_dense/kernel_1
�
Otransformer_layer_1/feedforward_intermediate_dense/kernel_1/Read/ReadVariableOpReadVariableOp;transformer_layer_1/feedforward_intermediate_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
&Variable_85/Initializer/ReadVariableOpReadVariableOp;transformer_layer_1/feedforward_intermediate_dense/kernel_1*
_class
loc:@Variable_85* 
_output_shapes
:
��*
dtype0
�
Variable_85VarHandleOp*
_class
loc:@Variable_85*
_output_shapes
: *

debug_nameVariable_85/*
dtype0*
shape:
��*
shared_nameVariable_85
g
,Variable_85/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_85*
_output_shapes
: 
h
Variable_85/AssignAssignVariableOpVariable_85&Variable_85/Initializer/ReadVariableOp*
dtype0
m
Variable_85/Read/ReadVariableOpReadVariableOpVariable_85* 
_output_shapes
:
��*
dtype0
�
1transformer_layer_1/feedforward_layer_norm/beta_1VarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_1/feedforward_layer_norm/beta_1/*
dtype0*
shape:�*B
shared_name31transformer_layer_1/feedforward_layer_norm/beta_1
�
Etransformer_layer_1/feedforward_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp1transformer_layer_1/feedforward_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_86/Initializer/ReadVariableOpReadVariableOp1transformer_layer_1/feedforward_layer_norm/beta_1*
_class
loc:@Variable_86*
_output_shapes	
:�*
dtype0
�
Variable_86VarHandleOp*
_class
loc:@Variable_86*
_output_shapes
: *

debug_nameVariable_86/*
dtype0*
shape:�*
shared_nameVariable_86
g
,Variable_86/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_86*
_output_shapes
: 
h
Variable_86/AssignAssignVariableOpVariable_86&Variable_86/Initializer/ReadVariableOp*
dtype0
h
Variable_86/Read/ReadVariableOpReadVariableOpVariable_86*
_output_shapes	
:�*
dtype0
�
2transformer_layer_1/feedforward_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_1/feedforward_layer_norm/gamma_1/*
dtype0*
shape:�*C
shared_name42transformer_layer_1/feedforward_layer_norm/gamma_1
�
Ftransformer_layer_1/feedforward_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp2transformer_layer_1/feedforward_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_87/Initializer/ReadVariableOpReadVariableOp2transformer_layer_1/feedforward_layer_norm/gamma_1*
_class
loc:@Variable_87*
_output_shapes	
:�*
dtype0
�
Variable_87VarHandleOp*
_class
loc:@Variable_87*
_output_shapes
: *

debug_nameVariable_87/*
dtype0*
shape:�*
shared_nameVariable_87
g
,Variable_87/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_87*
_output_shapes
: 
h
Variable_87/AssignAssignVariableOpVariable_87&Variable_87/Initializer/ReadVariableOp*
dtype0
h
Variable_87/Read/ReadVariableOpReadVariableOpVariable_87*
_output_shapes	
:�*
dtype0
�
*transformer_layer_1/seed_generator_state_1VarHandleOp*
_output_shapes
: *;

debug_name-+transformer_layer_1/seed_generator_state_1/*
dtype0*
shape:*;
shared_name,*transformer_layer_1/seed_generator_state_1
�
>transformer_layer_1/seed_generator_state_1/Read/ReadVariableOpReadVariableOp*transformer_layer_1/seed_generator_state_1*
_output_shapes
:*
dtype0
�
&Variable_88/Initializer/ReadVariableOpReadVariableOp*transformer_layer_1/seed_generator_state_1*
_class
loc:@Variable_88*
_output_shapes
:*
dtype0
�
Variable_88VarHandleOp*
_class
loc:@Variable_88*
_output_shapes
: *

debug_nameVariable_88/*
dtype0*
shape:*
shared_nameVariable_88
g
,Variable_88/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_88*
_output_shapes
: 
h
Variable_88/AssignAssignVariableOpVariable_88&Variable_88/Initializer/ReadVariableOp*
dtype0
g
Variable_88/Read/ReadVariableOpReadVariableOpVariable_88*
_output_shapes
:*
dtype0
�
4transformer_layer_1/self_attention_layer_norm/beta_1VarHandleOp*
_output_shapes
: *E

debug_name75transformer_layer_1/self_attention_layer_norm/beta_1/*
dtype0*
shape:�*E
shared_name64transformer_layer_1/self_attention_layer_norm/beta_1
�
Htransformer_layer_1/self_attention_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp4transformer_layer_1/self_attention_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_89/Initializer/ReadVariableOpReadVariableOp4transformer_layer_1/self_attention_layer_norm/beta_1*
_class
loc:@Variable_89*
_output_shapes	
:�*
dtype0
�
Variable_89VarHandleOp*
_class
loc:@Variable_89*
_output_shapes
: *

debug_nameVariable_89/*
dtype0*
shape:�*
shared_nameVariable_89
g
,Variable_89/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_89*
_output_shapes
: 
h
Variable_89/AssignAssignVariableOpVariable_89&Variable_89/Initializer/ReadVariableOp*
dtype0
h
Variable_89/Read/ReadVariableOpReadVariableOpVariable_89*
_output_shapes	
:�*
dtype0
�
5transformer_layer_1/self_attention_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_1/self_attention_layer_norm/gamma_1/*
dtype0*
shape:�*F
shared_name75transformer_layer_1/self_attention_layer_norm/gamma_1
�
Itransformer_layer_1/self_attention_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp5transformer_layer_1/self_attention_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_90/Initializer/ReadVariableOpReadVariableOp5transformer_layer_1/self_attention_layer_norm/gamma_1*
_class
loc:@Variable_90*
_output_shapes	
:�*
dtype0
�
Variable_90VarHandleOp*
_class
loc:@Variable_90*
_output_shapes
: *

debug_nameVariable_90/*
dtype0*
shape:�*
shared_nameVariable_90
g
,Variable_90/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_90*
_output_shapes
: 
h
Variable_90/AssignAssignVariableOpVariable_90&Variable_90/Initializer/ReadVariableOp*
dtype0
h
Variable_90/Read/ReadVariableOpReadVariableOpVariable_90*
_output_shapes	
:�*
dtype0
�
@transformer_layer_1/self_attention_layer/attention_output/bias_1VarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_1/self_attention_layer/attention_output/bias_1/*
dtype0*
shape:�*Q
shared_nameB@transformer_layer_1/self_attention_layer/attention_output/bias_1
�
Ttransformer_layer_1/self_attention_layer/attention_output/bias_1/Read/ReadVariableOpReadVariableOp@transformer_layer_1/self_attention_layer/attention_output/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_91/Initializer/ReadVariableOpReadVariableOp@transformer_layer_1/self_attention_layer/attention_output/bias_1*
_class
loc:@Variable_91*
_output_shapes	
:�*
dtype0
�
Variable_91VarHandleOp*
_class
loc:@Variable_91*
_output_shapes
: *

debug_nameVariable_91/*
dtype0*
shape:�*
shared_nameVariable_91
g
,Variable_91/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_91*
_output_shapes
: 
h
Variable_91/AssignAssignVariableOpVariable_91&Variable_91/Initializer/ReadVariableOp*
dtype0
h
Variable_91/Read/ReadVariableOpReadVariableOpVariable_91*
_output_shapes	
:�*
dtype0
�
Btransformer_layer_1/self_attention_layer/attention_output/kernel_1VarHandleOp*
_output_shapes
: *S

debug_nameECtransformer_layer_1/self_attention_layer/attention_output/kernel_1/*
dtype0*
shape:@�*S
shared_nameDBtransformer_layer_1/self_attention_layer/attention_output/kernel_1
�
Vtransformer_layer_1/self_attention_layer/attention_output/kernel_1/Read/ReadVariableOpReadVariableOpBtransformer_layer_1/self_attention_layer/attention_output/kernel_1*#
_output_shapes
:@�*
dtype0
�
&Variable_92/Initializer/ReadVariableOpReadVariableOpBtransformer_layer_1/self_attention_layer/attention_output/kernel_1*
_class
loc:@Variable_92*#
_output_shapes
:@�*
dtype0
�
Variable_92VarHandleOp*
_class
loc:@Variable_92*
_output_shapes
: *

debug_nameVariable_92/*
dtype0*
shape:@�*
shared_nameVariable_92
g
,Variable_92/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_92*
_output_shapes
: 
h
Variable_92/AssignAssignVariableOpVariable_92&Variable_92/Initializer/ReadVariableOp*
dtype0
p
Variable_92/Read/ReadVariableOpReadVariableOpVariable_92*#
_output_shapes
:@�*
dtype0
�
=transformer_layer_1/self_attention_layer/seed_generator_stateVarHandleOp*
_output_shapes
: *N

debug_name@>transformer_layer_1/self_attention_layer/seed_generator_state/*
dtype0*
shape:*N
shared_name?=transformer_layer_1/self_attention_layer/seed_generator_state
�
Qtransformer_layer_1/self_attention_layer/seed_generator_state/Read/ReadVariableOpReadVariableOp=transformer_layer_1/self_attention_layer/seed_generator_state*
_output_shapes
:*
dtype0
�
&Variable_93/Initializer/ReadVariableOpReadVariableOp=transformer_layer_1/self_attention_layer/seed_generator_state*
_class
loc:@Variable_93*
_output_shapes
:*
dtype0
�
Variable_93VarHandleOp*
_class
loc:@Variable_93*
_output_shapes
: *

debug_nameVariable_93/*
dtype0*
shape:*
shared_nameVariable_93
g
,Variable_93/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_93*
_output_shapes
: 
h
Variable_93/AssignAssignVariableOpVariable_93&Variable_93/Initializer/ReadVariableOp*
dtype0
g
Variable_93/Read/ReadVariableOpReadVariableOpVariable_93*
_output_shapes
:*
dtype0
�
5transformer_layer_1/self_attention_layer/value/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_1/self_attention_layer/value/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_1/self_attention_layer/value/bias_1
�
Itransformer_layer_1/self_attention_layer/value/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_1/self_attention_layer/value/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_94/Initializer/ReadVariableOpReadVariableOp5transformer_layer_1/self_attention_layer/value/bias_1*
_class
loc:@Variable_94*
_output_shapes

:@*
dtype0
�
Variable_94VarHandleOp*
_class
loc:@Variable_94*
_output_shapes
: *

debug_nameVariable_94/*
dtype0*
shape
:@*
shared_nameVariable_94
g
,Variable_94/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_94*
_output_shapes
: 
h
Variable_94/AssignAssignVariableOpVariable_94&Variable_94/Initializer/ReadVariableOp*
dtype0
k
Variable_94/Read/ReadVariableOpReadVariableOpVariable_94*
_output_shapes

:@*
dtype0
�
7transformer_layer_1/self_attention_layer/value/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_1/self_attention_layer/value/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_1/self_attention_layer/value/kernel_1
�
Ktransformer_layer_1/self_attention_layer/value/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_1/self_attention_layer/value/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_95/Initializer/ReadVariableOpReadVariableOp7transformer_layer_1/self_attention_layer/value/kernel_1*
_class
loc:@Variable_95*#
_output_shapes
:�@*
dtype0
�
Variable_95VarHandleOp*
_class
loc:@Variable_95*
_output_shapes
: *

debug_nameVariable_95/*
dtype0*
shape:�@*
shared_nameVariable_95
g
,Variable_95/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_95*
_output_shapes
: 
h
Variable_95/AssignAssignVariableOpVariable_95&Variable_95/Initializer/ReadVariableOp*
dtype0
p
Variable_95/Read/ReadVariableOpReadVariableOpVariable_95*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_1/self_attention_layer/key/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_1/self_attention_layer/key/bias_1/*
dtype0*
shape
:@*D
shared_name53transformer_layer_1/self_attention_layer/key/bias_1
�
Gtransformer_layer_1/self_attention_layer/key/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_1/self_attention_layer/key/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_96/Initializer/ReadVariableOpReadVariableOp3transformer_layer_1/self_attention_layer/key/bias_1*
_class
loc:@Variable_96*
_output_shapes

:@*
dtype0
�
Variable_96VarHandleOp*
_class
loc:@Variable_96*
_output_shapes
: *

debug_nameVariable_96/*
dtype0*
shape
:@*
shared_nameVariable_96
g
,Variable_96/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_96*
_output_shapes
: 
h
Variable_96/AssignAssignVariableOpVariable_96&Variable_96/Initializer/ReadVariableOp*
dtype0
k
Variable_96/Read/ReadVariableOpReadVariableOpVariable_96*
_output_shapes

:@*
dtype0
�
5transformer_layer_1/self_attention_layer/key/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_1/self_attention_layer/key/kernel_1/*
dtype0*
shape:�@*F
shared_name75transformer_layer_1/self_attention_layer/key/kernel_1
�
Itransformer_layer_1/self_attention_layer/key/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_1/self_attention_layer/key/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_97/Initializer/ReadVariableOpReadVariableOp5transformer_layer_1/self_attention_layer/key/kernel_1*
_class
loc:@Variable_97*#
_output_shapes
:�@*
dtype0
�
Variable_97VarHandleOp*
_class
loc:@Variable_97*
_output_shapes
: *

debug_nameVariable_97/*
dtype0*
shape:�@*
shared_nameVariable_97
g
,Variable_97/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_97*
_output_shapes
: 
h
Variable_97/AssignAssignVariableOpVariable_97&Variable_97/Initializer/ReadVariableOp*
dtype0
p
Variable_97/Read/ReadVariableOpReadVariableOpVariable_97*#
_output_shapes
:�@*
dtype0
�
5transformer_layer_1/self_attention_layer/query/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_1/self_attention_layer/query/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_1/self_attention_layer/query/bias_1
�
Itransformer_layer_1/self_attention_layer/query/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_1/self_attention_layer/query/bias_1*
_output_shapes

:@*
dtype0
�
&Variable_98/Initializer/ReadVariableOpReadVariableOp5transformer_layer_1/self_attention_layer/query/bias_1*
_class
loc:@Variable_98*
_output_shapes

:@*
dtype0
�
Variable_98VarHandleOp*
_class
loc:@Variable_98*
_output_shapes
: *

debug_nameVariable_98/*
dtype0*
shape
:@*
shared_nameVariable_98
g
,Variable_98/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_98*
_output_shapes
: 
h
Variable_98/AssignAssignVariableOpVariable_98&Variable_98/Initializer/ReadVariableOp*
dtype0
k
Variable_98/Read/ReadVariableOpReadVariableOpVariable_98*
_output_shapes

:@*
dtype0
�
7transformer_layer_1/self_attention_layer/query/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_1/self_attention_layer/query/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_1/self_attention_layer/query/kernel_1
�
Ktransformer_layer_1/self_attention_layer/query/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_1/self_attention_layer/query/kernel_1*#
_output_shapes
:�@*
dtype0
�
&Variable_99/Initializer/ReadVariableOpReadVariableOp7transformer_layer_1/self_attention_layer/query/kernel_1*
_class
loc:@Variable_99*#
_output_shapes
:�@*
dtype0
�
Variable_99VarHandleOp*
_class
loc:@Variable_99*
_output_shapes
: *

debug_nameVariable_99/*
dtype0*
shape:�@*
shared_nameVariable_99
g
,Variable_99/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_99*
_output_shapes
: 
h
Variable_99/AssignAssignVariableOpVariable_99&Variable_99/Initializer/ReadVariableOp*
dtype0
p
Variable_99/Read/ReadVariableOpReadVariableOpVariable_99*#
_output_shapes
:�@*
dtype0
�
(transformer_layer_0/seed_generator_stateVarHandleOp*
_output_shapes
: *9

debug_name+)transformer_layer_0/seed_generator_state/*
dtype0*
shape:*9
shared_name*(transformer_layer_0/seed_generator_state
�
<transformer_layer_0/seed_generator_state/Read/ReadVariableOpReadVariableOp(transformer_layer_0/seed_generator_state*
_output_shapes
:*
dtype0
�
'Variable_100/Initializer/ReadVariableOpReadVariableOp(transformer_layer_0/seed_generator_state*
_class
loc:@Variable_100*
_output_shapes
:*
dtype0
�
Variable_100VarHandleOp*
_class
loc:@Variable_100*
_output_shapes
: *

debug_nameVariable_100/*
dtype0*
shape:*
shared_nameVariable_100
i
-Variable_100/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_100*
_output_shapes
: 
k
Variable_100/AssignAssignVariableOpVariable_100'Variable_100/Initializer/ReadVariableOp*
dtype0
i
 Variable_100/Read/ReadVariableOpReadVariableOpVariable_100*
_output_shapes
:*
dtype0
�
3transformer_layer_0/feedforward_output_dense/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_0/feedforward_output_dense/bias_1/*
dtype0*
shape:�*D
shared_name53transformer_layer_0/feedforward_output_dense/bias_1
�
Gtransformer_layer_0/feedforward_output_dense/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_0/feedforward_output_dense/bias_1*
_output_shapes	
:�*
dtype0
�
'Variable_101/Initializer/ReadVariableOpReadVariableOp3transformer_layer_0/feedforward_output_dense/bias_1*
_class
loc:@Variable_101*
_output_shapes	
:�*
dtype0
�
Variable_101VarHandleOp*
_class
loc:@Variable_101*
_output_shapes
: *

debug_nameVariable_101/*
dtype0*
shape:�*
shared_nameVariable_101
i
-Variable_101/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_101*
_output_shapes
: 
k
Variable_101/AssignAssignVariableOpVariable_101'Variable_101/Initializer/ReadVariableOp*
dtype0
j
 Variable_101/Read/ReadVariableOpReadVariableOpVariable_101*
_output_shapes	
:�*
dtype0
�
5transformer_layer_0/feedforward_output_dense/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_0/feedforward_output_dense/kernel_1/*
dtype0*
shape:
��*F
shared_name75transformer_layer_0/feedforward_output_dense/kernel_1
�
Itransformer_layer_0/feedforward_output_dense/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_0/feedforward_output_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
'Variable_102/Initializer/ReadVariableOpReadVariableOp5transformer_layer_0/feedforward_output_dense/kernel_1*
_class
loc:@Variable_102* 
_output_shapes
:
��*
dtype0
�
Variable_102VarHandleOp*
_class
loc:@Variable_102*
_output_shapes
: *

debug_nameVariable_102/*
dtype0*
shape:
��*
shared_nameVariable_102
i
-Variable_102/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_102*
_output_shapes
: 
k
Variable_102/AssignAssignVariableOpVariable_102'Variable_102/Initializer/ReadVariableOp*
dtype0
o
 Variable_102/Read/ReadVariableOpReadVariableOpVariable_102* 
_output_shapes
:
��*
dtype0
�
9transformer_layer_0/feedforward_intermediate_dense/bias_1VarHandleOp*
_output_shapes
: *J

debug_name<:transformer_layer_0/feedforward_intermediate_dense/bias_1/*
dtype0*
shape:�*J
shared_name;9transformer_layer_0/feedforward_intermediate_dense/bias_1
�
Mtransformer_layer_0/feedforward_intermediate_dense/bias_1/Read/ReadVariableOpReadVariableOp9transformer_layer_0/feedforward_intermediate_dense/bias_1*
_output_shapes	
:�*
dtype0
�
'Variable_103/Initializer/ReadVariableOpReadVariableOp9transformer_layer_0/feedforward_intermediate_dense/bias_1*
_class
loc:@Variable_103*
_output_shapes	
:�*
dtype0
�
Variable_103VarHandleOp*
_class
loc:@Variable_103*
_output_shapes
: *

debug_nameVariable_103/*
dtype0*
shape:�*
shared_nameVariable_103
i
-Variable_103/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_103*
_output_shapes
: 
k
Variable_103/AssignAssignVariableOpVariable_103'Variable_103/Initializer/ReadVariableOp*
dtype0
j
 Variable_103/Read/ReadVariableOpReadVariableOpVariable_103*
_output_shapes	
:�*
dtype0
�
;transformer_layer_0/feedforward_intermediate_dense/kernel_1VarHandleOp*
_output_shapes
: *L

debug_name><transformer_layer_0/feedforward_intermediate_dense/kernel_1/*
dtype0*
shape:
��*L
shared_name=;transformer_layer_0/feedforward_intermediate_dense/kernel_1
�
Otransformer_layer_0/feedforward_intermediate_dense/kernel_1/Read/ReadVariableOpReadVariableOp;transformer_layer_0/feedforward_intermediate_dense/kernel_1* 
_output_shapes
:
��*
dtype0
�
'Variable_104/Initializer/ReadVariableOpReadVariableOp;transformer_layer_0/feedforward_intermediate_dense/kernel_1*
_class
loc:@Variable_104* 
_output_shapes
:
��*
dtype0
�
Variable_104VarHandleOp*
_class
loc:@Variable_104*
_output_shapes
: *

debug_nameVariable_104/*
dtype0*
shape:
��*
shared_nameVariable_104
i
-Variable_104/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_104*
_output_shapes
: 
k
Variable_104/AssignAssignVariableOpVariable_104'Variable_104/Initializer/ReadVariableOp*
dtype0
o
 Variable_104/Read/ReadVariableOpReadVariableOpVariable_104* 
_output_shapes
:
��*
dtype0
�
1transformer_layer_0/feedforward_layer_norm/beta_1VarHandleOp*
_output_shapes
: *B

debug_name42transformer_layer_0/feedforward_layer_norm/beta_1/*
dtype0*
shape:�*B
shared_name31transformer_layer_0/feedforward_layer_norm/beta_1
�
Etransformer_layer_0/feedforward_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp1transformer_layer_0/feedforward_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
'Variable_105/Initializer/ReadVariableOpReadVariableOp1transformer_layer_0/feedforward_layer_norm/beta_1*
_class
loc:@Variable_105*
_output_shapes	
:�*
dtype0
�
Variable_105VarHandleOp*
_class
loc:@Variable_105*
_output_shapes
: *

debug_nameVariable_105/*
dtype0*
shape:�*
shared_nameVariable_105
i
-Variable_105/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_105*
_output_shapes
: 
k
Variable_105/AssignAssignVariableOpVariable_105'Variable_105/Initializer/ReadVariableOp*
dtype0
j
 Variable_105/Read/ReadVariableOpReadVariableOpVariable_105*
_output_shapes	
:�*
dtype0
�
2transformer_layer_0/feedforward_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *C

debug_name53transformer_layer_0/feedforward_layer_norm/gamma_1/*
dtype0*
shape:�*C
shared_name42transformer_layer_0/feedforward_layer_norm/gamma_1
�
Ftransformer_layer_0/feedforward_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp2transformer_layer_0/feedforward_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
'Variable_106/Initializer/ReadVariableOpReadVariableOp2transformer_layer_0/feedforward_layer_norm/gamma_1*
_class
loc:@Variable_106*
_output_shapes	
:�*
dtype0
�
Variable_106VarHandleOp*
_class
loc:@Variable_106*
_output_shapes
: *

debug_nameVariable_106/*
dtype0*
shape:�*
shared_nameVariable_106
i
-Variable_106/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_106*
_output_shapes
: 
k
Variable_106/AssignAssignVariableOpVariable_106'Variable_106/Initializer/ReadVariableOp*
dtype0
j
 Variable_106/Read/ReadVariableOpReadVariableOpVariable_106*
_output_shapes	
:�*
dtype0
�
*transformer_layer_0/seed_generator_state_1VarHandleOp*
_output_shapes
: *;

debug_name-+transformer_layer_0/seed_generator_state_1/*
dtype0*
shape:*;
shared_name,*transformer_layer_0/seed_generator_state_1
�
>transformer_layer_0/seed_generator_state_1/Read/ReadVariableOpReadVariableOp*transformer_layer_0/seed_generator_state_1*
_output_shapes
:*
dtype0
�
'Variable_107/Initializer/ReadVariableOpReadVariableOp*transformer_layer_0/seed_generator_state_1*
_class
loc:@Variable_107*
_output_shapes
:*
dtype0
�
Variable_107VarHandleOp*
_class
loc:@Variable_107*
_output_shapes
: *

debug_nameVariable_107/*
dtype0*
shape:*
shared_nameVariable_107
i
-Variable_107/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_107*
_output_shapes
: 
k
Variable_107/AssignAssignVariableOpVariable_107'Variable_107/Initializer/ReadVariableOp*
dtype0
i
 Variable_107/Read/ReadVariableOpReadVariableOpVariable_107*
_output_shapes
:*
dtype0
�
4transformer_layer_0/self_attention_layer_norm/beta_1VarHandleOp*
_output_shapes
: *E

debug_name75transformer_layer_0/self_attention_layer_norm/beta_1/*
dtype0*
shape:�*E
shared_name64transformer_layer_0/self_attention_layer_norm/beta_1
�
Htransformer_layer_0/self_attention_layer_norm/beta_1/Read/ReadVariableOpReadVariableOp4transformer_layer_0/self_attention_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
'Variable_108/Initializer/ReadVariableOpReadVariableOp4transformer_layer_0/self_attention_layer_norm/beta_1*
_class
loc:@Variable_108*
_output_shapes	
:�*
dtype0
�
Variable_108VarHandleOp*
_class
loc:@Variable_108*
_output_shapes
: *

debug_nameVariable_108/*
dtype0*
shape:�*
shared_nameVariable_108
i
-Variable_108/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_108*
_output_shapes
: 
k
Variable_108/AssignAssignVariableOpVariable_108'Variable_108/Initializer/ReadVariableOp*
dtype0
j
 Variable_108/Read/ReadVariableOpReadVariableOpVariable_108*
_output_shapes	
:�*
dtype0
�
5transformer_layer_0/self_attention_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_0/self_attention_layer_norm/gamma_1/*
dtype0*
shape:�*F
shared_name75transformer_layer_0/self_attention_layer_norm/gamma_1
�
Itransformer_layer_0/self_attention_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOp5transformer_layer_0/self_attention_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
'Variable_109/Initializer/ReadVariableOpReadVariableOp5transformer_layer_0/self_attention_layer_norm/gamma_1*
_class
loc:@Variable_109*
_output_shapes	
:�*
dtype0
�
Variable_109VarHandleOp*
_class
loc:@Variable_109*
_output_shapes
: *

debug_nameVariable_109/*
dtype0*
shape:�*
shared_nameVariable_109
i
-Variable_109/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_109*
_output_shapes
: 
k
Variable_109/AssignAssignVariableOpVariable_109'Variable_109/Initializer/ReadVariableOp*
dtype0
j
 Variable_109/Read/ReadVariableOpReadVariableOpVariable_109*
_output_shapes	
:�*
dtype0
�
@transformer_layer_0/self_attention_layer/attention_output/bias_1VarHandleOp*
_output_shapes
: *Q

debug_nameCAtransformer_layer_0/self_attention_layer/attention_output/bias_1/*
dtype0*
shape:�*Q
shared_nameB@transformer_layer_0/self_attention_layer/attention_output/bias_1
�
Ttransformer_layer_0/self_attention_layer/attention_output/bias_1/Read/ReadVariableOpReadVariableOp@transformer_layer_0/self_attention_layer/attention_output/bias_1*
_output_shapes	
:�*
dtype0
�
'Variable_110/Initializer/ReadVariableOpReadVariableOp@transformer_layer_0/self_attention_layer/attention_output/bias_1*
_class
loc:@Variable_110*
_output_shapes	
:�*
dtype0
�
Variable_110VarHandleOp*
_class
loc:@Variable_110*
_output_shapes
: *

debug_nameVariable_110/*
dtype0*
shape:�*
shared_nameVariable_110
i
-Variable_110/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_110*
_output_shapes
: 
k
Variable_110/AssignAssignVariableOpVariable_110'Variable_110/Initializer/ReadVariableOp*
dtype0
j
 Variable_110/Read/ReadVariableOpReadVariableOpVariable_110*
_output_shapes	
:�*
dtype0
�
Btransformer_layer_0/self_attention_layer/attention_output/kernel_1VarHandleOp*
_output_shapes
: *S

debug_nameECtransformer_layer_0/self_attention_layer/attention_output/kernel_1/*
dtype0*
shape:@�*S
shared_nameDBtransformer_layer_0/self_attention_layer/attention_output/kernel_1
�
Vtransformer_layer_0/self_attention_layer/attention_output/kernel_1/Read/ReadVariableOpReadVariableOpBtransformer_layer_0/self_attention_layer/attention_output/kernel_1*#
_output_shapes
:@�*
dtype0
�
'Variable_111/Initializer/ReadVariableOpReadVariableOpBtransformer_layer_0/self_attention_layer/attention_output/kernel_1*
_class
loc:@Variable_111*#
_output_shapes
:@�*
dtype0
�
Variable_111VarHandleOp*
_class
loc:@Variable_111*
_output_shapes
: *

debug_nameVariable_111/*
dtype0*
shape:@�*
shared_nameVariable_111
i
-Variable_111/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_111*
_output_shapes
: 
k
Variable_111/AssignAssignVariableOpVariable_111'Variable_111/Initializer/ReadVariableOp*
dtype0
r
 Variable_111/Read/ReadVariableOpReadVariableOpVariable_111*#
_output_shapes
:@�*
dtype0
�
=transformer_layer_0/self_attention_layer/seed_generator_stateVarHandleOp*
_output_shapes
: *N

debug_name@>transformer_layer_0/self_attention_layer/seed_generator_state/*
dtype0*
shape:*N
shared_name?=transformer_layer_0/self_attention_layer/seed_generator_state
�
Qtransformer_layer_0/self_attention_layer/seed_generator_state/Read/ReadVariableOpReadVariableOp=transformer_layer_0/self_attention_layer/seed_generator_state*
_output_shapes
:*
dtype0
�
'Variable_112/Initializer/ReadVariableOpReadVariableOp=transformer_layer_0/self_attention_layer/seed_generator_state*
_class
loc:@Variable_112*
_output_shapes
:*
dtype0
�
Variable_112VarHandleOp*
_class
loc:@Variable_112*
_output_shapes
: *

debug_nameVariable_112/*
dtype0*
shape:*
shared_nameVariable_112
i
-Variable_112/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_112*
_output_shapes
: 
k
Variable_112/AssignAssignVariableOpVariable_112'Variable_112/Initializer/ReadVariableOp*
dtype0
i
 Variable_112/Read/ReadVariableOpReadVariableOpVariable_112*
_output_shapes
:*
dtype0
�
5transformer_layer_0/self_attention_layer/value/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_0/self_attention_layer/value/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_0/self_attention_layer/value/bias_1
�
Itransformer_layer_0/self_attention_layer/value/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_0/self_attention_layer/value/bias_1*
_output_shapes

:@*
dtype0
�
'Variable_113/Initializer/ReadVariableOpReadVariableOp5transformer_layer_0/self_attention_layer/value/bias_1*
_class
loc:@Variable_113*
_output_shapes

:@*
dtype0
�
Variable_113VarHandleOp*
_class
loc:@Variable_113*
_output_shapes
: *

debug_nameVariable_113/*
dtype0*
shape
:@*
shared_nameVariable_113
i
-Variable_113/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_113*
_output_shapes
: 
k
Variable_113/AssignAssignVariableOpVariable_113'Variable_113/Initializer/ReadVariableOp*
dtype0
m
 Variable_113/Read/ReadVariableOpReadVariableOpVariable_113*
_output_shapes

:@*
dtype0
�
7transformer_layer_0/self_attention_layer/value/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_0/self_attention_layer/value/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_0/self_attention_layer/value/kernel_1
�
Ktransformer_layer_0/self_attention_layer/value/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_0/self_attention_layer/value/kernel_1*#
_output_shapes
:�@*
dtype0
�
'Variable_114/Initializer/ReadVariableOpReadVariableOp7transformer_layer_0/self_attention_layer/value/kernel_1*
_class
loc:@Variable_114*#
_output_shapes
:�@*
dtype0
�
Variable_114VarHandleOp*
_class
loc:@Variable_114*
_output_shapes
: *

debug_nameVariable_114/*
dtype0*
shape:�@*
shared_nameVariable_114
i
-Variable_114/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_114*
_output_shapes
: 
k
Variable_114/AssignAssignVariableOpVariable_114'Variable_114/Initializer/ReadVariableOp*
dtype0
r
 Variable_114/Read/ReadVariableOpReadVariableOpVariable_114*#
_output_shapes
:�@*
dtype0
�
3transformer_layer_0/self_attention_layer/key/bias_1VarHandleOp*
_output_shapes
: *D

debug_name64transformer_layer_0/self_attention_layer/key/bias_1/*
dtype0*
shape
:@*D
shared_name53transformer_layer_0/self_attention_layer/key/bias_1
�
Gtransformer_layer_0/self_attention_layer/key/bias_1/Read/ReadVariableOpReadVariableOp3transformer_layer_0/self_attention_layer/key/bias_1*
_output_shapes

:@*
dtype0
�
'Variable_115/Initializer/ReadVariableOpReadVariableOp3transformer_layer_0/self_attention_layer/key/bias_1*
_class
loc:@Variable_115*
_output_shapes

:@*
dtype0
�
Variable_115VarHandleOp*
_class
loc:@Variable_115*
_output_shapes
: *

debug_nameVariable_115/*
dtype0*
shape
:@*
shared_nameVariable_115
i
-Variable_115/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_115*
_output_shapes
: 
k
Variable_115/AssignAssignVariableOpVariable_115'Variable_115/Initializer/ReadVariableOp*
dtype0
m
 Variable_115/Read/ReadVariableOpReadVariableOpVariable_115*
_output_shapes

:@*
dtype0
�
5transformer_layer_0/self_attention_layer/key/kernel_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_0/self_attention_layer/key/kernel_1/*
dtype0*
shape:�@*F
shared_name75transformer_layer_0/self_attention_layer/key/kernel_1
�
Itransformer_layer_0/self_attention_layer/key/kernel_1/Read/ReadVariableOpReadVariableOp5transformer_layer_0/self_attention_layer/key/kernel_1*#
_output_shapes
:�@*
dtype0
�
'Variable_116/Initializer/ReadVariableOpReadVariableOp5transformer_layer_0/self_attention_layer/key/kernel_1*
_class
loc:@Variable_116*#
_output_shapes
:�@*
dtype0
�
Variable_116VarHandleOp*
_class
loc:@Variable_116*
_output_shapes
: *

debug_nameVariable_116/*
dtype0*
shape:�@*
shared_nameVariable_116
i
-Variable_116/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_116*
_output_shapes
: 
k
Variable_116/AssignAssignVariableOpVariable_116'Variable_116/Initializer/ReadVariableOp*
dtype0
r
 Variable_116/Read/ReadVariableOpReadVariableOpVariable_116*#
_output_shapes
:�@*
dtype0
�
5transformer_layer_0/self_attention_layer/query/bias_1VarHandleOp*
_output_shapes
: *F

debug_name86transformer_layer_0/self_attention_layer/query/bias_1/*
dtype0*
shape
:@*F
shared_name75transformer_layer_0/self_attention_layer/query/bias_1
�
Itransformer_layer_0/self_attention_layer/query/bias_1/Read/ReadVariableOpReadVariableOp5transformer_layer_0/self_attention_layer/query/bias_1*
_output_shapes

:@*
dtype0
�
'Variable_117/Initializer/ReadVariableOpReadVariableOp5transformer_layer_0/self_attention_layer/query/bias_1*
_class
loc:@Variable_117*
_output_shapes

:@*
dtype0
�
Variable_117VarHandleOp*
_class
loc:@Variable_117*
_output_shapes
: *

debug_nameVariable_117/*
dtype0*
shape
:@*
shared_nameVariable_117
i
-Variable_117/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_117*
_output_shapes
: 
k
Variable_117/AssignAssignVariableOpVariable_117'Variable_117/Initializer/ReadVariableOp*
dtype0
m
 Variable_117/Read/ReadVariableOpReadVariableOpVariable_117*
_output_shapes

:@*
dtype0
�
7transformer_layer_0/self_attention_layer/query/kernel_1VarHandleOp*
_output_shapes
: *H

debug_name:8transformer_layer_0/self_attention_layer/query/kernel_1/*
dtype0*
shape:�@*H
shared_name97transformer_layer_0/self_attention_layer/query/kernel_1
�
Ktransformer_layer_0/self_attention_layer/query/kernel_1/Read/ReadVariableOpReadVariableOp7transformer_layer_0/self_attention_layer/query/kernel_1*#
_output_shapes
:�@*
dtype0
�
'Variable_118/Initializer/ReadVariableOpReadVariableOp7transformer_layer_0/self_attention_layer/query/kernel_1*
_class
loc:@Variable_118*#
_output_shapes
:�@*
dtype0
�
Variable_118VarHandleOp*
_class
loc:@Variable_118*
_output_shapes
: *

debug_nameVariable_118/*
dtype0*
shape:�@*
shared_nameVariable_118
i
-Variable_118/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_118*
_output_shapes
: 
k
Variable_118/AssignAssignVariableOpVariable_118'Variable_118/Initializer/ReadVariableOp*
dtype0
r
 Variable_118/Read/ReadVariableOpReadVariableOpVariable_118*#
_output_shapes
:�@*
dtype0
�
seed_generator_state_1VarHandleOp*
_output_shapes
: *'

debug_nameseed_generator_state_1/*
dtype0*
shape:*'
shared_nameseed_generator_state_1
}
*seed_generator_state_1/Read/ReadVariableOpReadVariableOpseed_generator_state_1*
_output_shapes
:*
dtype0
�
'Variable_119/Initializer/ReadVariableOpReadVariableOpseed_generator_state_1*
_class
loc:@Variable_119*
_output_shapes
:*
dtype0
�
Variable_119VarHandleOp*
_class
loc:@Variable_119*
_output_shapes
: *

debug_nameVariable_119/*
dtype0*
shape:*
shared_nameVariable_119
i
-Variable_119/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_119*
_output_shapes
: 
k
Variable_119/AssignAssignVariableOpVariable_119'Variable_119/Initializer/ReadVariableOp*
dtype0
i
 Variable_119/Read/ReadVariableOpReadVariableOpVariable_119*
_output_shapes
:*
dtype0
�
embeddings_layer_norm/beta_1VarHandleOp*
_output_shapes
: *-

debug_nameembeddings_layer_norm/beta_1/*
dtype0*
shape:�*-
shared_nameembeddings_layer_norm/beta_1
�
0embeddings_layer_norm/beta_1/Read/ReadVariableOpReadVariableOpembeddings_layer_norm/beta_1*
_output_shapes	
:�*
dtype0
�
'Variable_120/Initializer/ReadVariableOpReadVariableOpembeddings_layer_norm/beta_1*
_class
loc:@Variable_120*
_output_shapes	
:�*
dtype0
�
Variable_120VarHandleOp*
_class
loc:@Variable_120*
_output_shapes
: *

debug_nameVariable_120/*
dtype0*
shape:�*
shared_nameVariable_120
i
-Variable_120/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_120*
_output_shapes
: 
k
Variable_120/AssignAssignVariableOpVariable_120'Variable_120/Initializer/ReadVariableOp*
dtype0
j
 Variable_120/Read/ReadVariableOpReadVariableOpVariable_120*
_output_shapes	
:�*
dtype0
�
embeddings_layer_norm/gamma_1VarHandleOp*
_output_shapes
: *.

debug_name embeddings_layer_norm/gamma_1/*
dtype0*
shape:�*.
shared_nameembeddings_layer_norm/gamma_1
�
1embeddings_layer_norm/gamma_1/Read/ReadVariableOpReadVariableOpembeddings_layer_norm/gamma_1*
_output_shapes	
:�*
dtype0
�
'Variable_121/Initializer/ReadVariableOpReadVariableOpembeddings_layer_norm/gamma_1*
_class
loc:@Variable_121*
_output_shapes	
:�*
dtype0
�
Variable_121VarHandleOp*
_class
loc:@Variable_121*
_output_shapes
: *

debug_nameVariable_121/*
dtype0*
shape:�*
shared_nameVariable_121
i
-Variable_121/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_121*
_output_shapes
: 
k
Variable_121/AssignAssignVariableOpVariable_121'Variable_121/Initializer/ReadVariableOp*
dtype0
j
 Variable_121/Read/ReadVariableOpReadVariableOpVariable_121*
_output_shapes	
:�*
dtype0
�
<token_and_position_embedding/position_embedding/embeddings_1VarHandleOp*
_output_shapes
: *M

debug_name?=token_and_position_embedding/position_embedding/embeddings_1/*
dtype0*
shape:
��*M
shared_name><token_and_position_embedding/position_embedding/embeddings_1
�
Ptoken_and_position_embedding/position_embedding/embeddings_1/Read/ReadVariableOpReadVariableOp<token_and_position_embedding/position_embedding/embeddings_1* 
_output_shapes
:
��*
dtype0
�
'Variable_122/Initializer/ReadVariableOpReadVariableOp<token_and_position_embedding/position_embedding/embeddings_1*
_class
loc:@Variable_122* 
_output_shapes
:
��*
dtype0
�
Variable_122VarHandleOp*
_class
loc:@Variable_122*
_output_shapes
: *

debug_nameVariable_122/*
dtype0*
shape:
��*
shared_nameVariable_122
i
-Variable_122/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_122*
_output_shapes
: 
k
Variable_122/AssignAssignVariableOpVariable_122'Variable_122/Initializer/ReadVariableOp*
dtype0
o
 Variable_122/Read/ReadVariableOpReadVariableOpVariable_122* 
_output_shapes
:
��*
dtype0
�
9token_and_position_embedding/token_embedding/embeddings_1VarHandleOp*
_output_shapes
: *J

debug_name<:token_and_position_embedding/token_embedding/embeddings_1/*
dtype0*
shape:���*J
shared_name;9token_and_position_embedding/token_embedding/embeddings_1
�
Mtoken_and_position_embedding/token_embedding/embeddings_1/Read/ReadVariableOpReadVariableOp9token_and_position_embedding/token_embedding/embeddings_1*!
_output_shapes
:���*
dtype0
�
'Variable_123/Initializer/ReadVariableOpReadVariableOp9token_and_position_embedding/token_embedding/embeddings_1*
_class
loc:@Variable_123*!
_output_shapes
:���*
dtype0
�
Variable_123VarHandleOp*
_class
loc:@Variable_123*
_output_shapes
: *

debug_nameVariable_123/*
dtype0*
shape:���*
shared_nameVariable_123
i
-Variable_123/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_123*
_output_shapes
: 
k
Variable_123/AssignAssignVariableOpVariable_123'Variable_123/Initializer/ReadVariableOp*
dtype0
p
 Variable_123/Read/ReadVariableOpReadVariableOpVariable_123*!
_output_shapes
:���*
dtype0
�
serve_padding_maskPlaceholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�
serve_token_idsPlaceholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�0
StatefulPartitionedCallStatefulPartitionedCallserve_padding_maskserve_token_ids9token_and_position_embedding/token_embedding/embeddings_1<token_and_position_embedding/position_embedding/embeddings_1embeddings_layer_norm/gamma_1embeddings_layer_norm/beta_17transformer_layer_0/self_attention_layer/query/kernel_15transformer_layer_0/self_attention_layer/query/bias_15transformer_layer_0/self_attention_layer/key/kernel_13transformer_layer_0/self_attention_layer/key/bias_17transformer_layer_0/self_attention_layer/value/kernel_15transformer_layer_0/self_attention_layer/value/bias_1Btransformer_layer_0/self_attention_layer/attention_output/kernel_1@transformer_layer_0/self_attention_layer/attention_output/bias_15transformer_layer_0/self_attention_layer_norm/gamma_14transformer_layer_0/self_attention_layer_norm/beta_1;transformer_layer_0/feedforward_intermediate_dense/kernel_19transformer_layer_0/feedforward_intermediate_dense/bias_15transformer_layer_0/feedforward_output_dense/kernel_13transformer_layer_0/feedforward_output_dense/bias_12transformer_layer_0/feedforward_layer_norm/gamma_11transformer_layer_0/feedforward_layer_norm/beta_17transformer_layer_1/self_attention_layer/query/kernel_15transformer_layer_1/self_attention_layer/query/bias_15transformer_layer_1/self_attention_layer/key/kernel_13transformer_layer_1/self_attention_layer/key/bias_17transformer_layer_1/self_attention_layer/value/kernel_15transformer_layer_1/self_attention_layer/value/bias_1Btransformer_layer_1/self_attention_layer/attention_output/kernel_1@transformer_layer_1/self_attention_layer/attention_output/bias_15transformer_layer_1/self_attention_layer_norm/gamma_14transformer_layer_1/self_attention_layer_norm/beta_1;transformer_layer_1/feedforward_intermediate_dense/kernel_19transformer_layer_1/feedforward_intermediate_dense/bias_15transformer_layer_1/feedforward_output_dense/kernel_13transformer_layer_1/feedforward_output_dense/bias_12transformer_layer_1/feedforward_layer_norm/gamma_11transformer_layer_1/feedforward_layer_norm/beta_17transformer_layer_2/self_attention_layer/query/kernel_15transformer_layer_2/self_attention_layer/query/bias_15transformer_layer_2/self_attention_layer/key/kernel_13transformer_layer_2/self_attention_layer/key/bias_17transformer_layer_2/self_attention_layer/value/kernel_15transformer_layer_2/self_attention_layer/value/bias_1Btransformer_layer_2/self_attention_layer/attention_output/kernel_1@transformer_layer_2/self_attention_layer/attention_output/bias_15transformer_layer_2/self_attention_layer_norm/gamma_14transformer_layer_2/self_attention_layer_norm/beta_1;transformer_layer_2/feedforward_intermediate_dense/kernel_19transformer_layer_2/feedforward_intermediate_dense/bias_15transformer_layer_2/feedforward_output_dense/kernel_13transformer_layer_2/feedforward_output_dense/bias_12transformer_layer_2/feedforward_layer_norm/gamma_11transformer_layer_2/feedforward_layer_norm/beta_17transformer_layer_3/self_attention_layer/query/kernel_15transformer_layer_3/self_attention_layer/query/bias_15transformer_layer_3/self_attention_layer/key/kernel_13transformer_layer_3/self_attention_layer/key/bias_17transformer_layer_3/self_attention_layer/value/kernel_15transformer_layer_3/self_attention_layer/value/bias_1Btransformer_layer_3/self_attention_layer/attention_output/kernel_1@transformer_layer_3/self_attention_layer/attention_output/bias_15transformer_layer_3/self_attention_layer_norm/gamma_14transformer_layer_3/self_attention_layer_norm/beta_1;transformer_layer_3/feedforward_intermediate_dense/kernel_19transformer_layer_3/feedforward_intermediate_dense/bias_15transformer_layer_3/feedforward_output_dense/kernel_13transformer_layer_3/feedforward_output_dense/bias_12transformer_layer_3/feedforward_layer_norm/gamma_11transformer_layer_3/feedforward_layer_norm/beta_17transformer_layer_4/self_attention_layer/query/kernel_15transformer_layer_4/self_attention_layer/query/bias_15transformer_layer_4/self_attention_layer/key/kernel_13transformer_layer_4/self_attention_layer/key/bias_17transformer_layer_4/self_attention_layer/value/kernel_15transformer_layer_4/self_attention_layer/value/bias_1Btransformer_layer_4/self_attention_layer/attention_output/kernel_1@transformer_layer_4/self_attention_layer/attention_output/bias_15transformer_layer_4/self_attention_layer_norm/gamma_14transformer_layer_4/self_attention_layer_norm/beta_1;transformer_layer_4/feedforward_intermediate_dense/kernel_19transformer_layer_4/feedforward_intermediate_dense/bias_15transformer_layer_4/feedforward_output_dense/kernel_13transformer_layer_4/feedforward_output_dense/bias_12transformer_layer_4/feedforward_layer_norm/gamma_11transformer_layer_4/feedforward_layer_norm/beta_17transformer_layer_5/self_attention_layer/query/kernel_15transformer_layer_5/self_attention_layer/query/bias_15transformer_layer_5/self_attention_layer/key/kernel_13transformer_layer_5/self_attention_layer/key/bias_17transformer_layer_5/self_attention_layer/value/kernel_15transformer_layer_5/self_attention_layer/value/bias_1Btransformer_layer_5/self_attention_layer/attention_output/kernel_1@transformer_layer_5/self_attention_layer/attention_output/bias_15transformer_layer_5/self_attention_layer_norm/gamma_14transformer_layer_5/self_attention_layer_norm/beta_1;transformer_layer_5/feedforward_intermediate_dense/kernel_19transformer_layer_5/feedforward_intermediate_dense/bias_15transformer_layer_5/feedforward_output_dense/kernel_13transformer_layer_5/feedforward_output_dense/bias_12transformer_layer_5/feedforward_layer_norm/gamma_11transformer_layer_5/feedforward_layer_norm/beta_1pooled_dense/kernel_1pooled_dense/bias_1logits/kernel_1logits/bias_1*u
Tinn
l2j*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*�
_read_only_resource_inputsl
jh	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghi*-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_signature_wrapper___call___8311
�
serving_default_padding_maskPlaceholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�
serving_default_token_idsPlaceholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�0
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_padding_maskserving_default_token_ids9token_and_position_embedding/token_embedding/embeddings_1<token_and_position_embedding/position_embedding/embeddings_1embeddings_layer_norm/gamma_1embeddings_layer_norm/beta_17transformer_layer_0/self_attention_layer/query/kernel_15transformer_layer_0/self_attention_layer/query/bias_15transformer_layer_0/self_attention_layer/key/kernel_13transformer_layer_0/self_attention_layer/key/bias_17transformer_layer_0/self_attention_layer/value/kernel_15transformer_layer_0/self_attention_layer/value/bias_1Btransformer_layer_0/self_attention_layer/attention_output/kernel_1@transformer_layer_0/self_attention_layer/attention_output/bias_15transformer_layer_0/self_attention_layer_norm/gamma_14transformer_layer_0/self_attention_layer_norm/beta_1;transformer_layer_0/feedforward_intermediate_dense/kernel_19transformer_layer_0/feedforward_intermediate_dense/bias_15transformer_layer_0/feedforward_output_dense/kernel_13transformer_layer_0/feedforward_output_dense/bias_12transformer_layer_0/feedforward_layer_norm/gamma_11transformer_layer_0/feedforward_layer_norm/beta_17transformer_layer_1/self_attention_layer/query/kernel_15transformer_layer_1/self_attention_layer/query/bias_15transformer_layer_1/self_attention_layer/key/kernel_13transformer_layer_1/self_attention_layer/key/bias_17transformer_layer_1/self_attention_layer/value/kernel_15transformer_layer_1/self_attention_layer/value/bias_1Btransformer_layer_1/self_attention_layer/attention_output/kernel_1@transformer_layer_1/self_attention_layer/attention_output/bias_15transformer_layer_1/self_attention_layer_norm/gamma_14transformer_layer_1/self_attention_layer_norm/beta_1;transformer_layer_1/feedforward_intermediate_dense/kernel_19transformer_layer_1/feedforward_intermediate_dense/bias_15transformer_layer_1/feedforward_output_dense/kernel_13transformer_layer_1/feedforward_output_dense/bias_12transformer_layer_1/feedforward_layer_norm/gamma_11transformer_layer_1/feedforward_layer_norm/beta_17transformer_layer_2/self_attention_layer/query/kernel_15transformer_layer_2/self_attention_layer/query/bias_15transformer_layer_2/self_attention_layer/key/kernel_13transformer_layer_2/self_attention_layer/key/bias_17transformer_layer_2/self_attention_layer/value/kernel_15transformer_layer_2/self_attention_layer/value/bias_1Btransformer_layer_2/self_attention_layer/attention_output/kernel_1@transformer_layer_2/self_attention_layer/attention_output/bias_15transformer_layer_2/self_attention_layer_norm/gamma_14transformer_layer_2/self_attention_layer_norm/beta_1;transformer_layer_2/feedforward_intermediate_dense/kernel_19transformer_layer_2/feedforward_intermediate_dense/bias_15transformer_layer_2/feedforward_output_dense/kernel_13transformer_layer_2/feedforward_output_dense/bias_12transformer_layer_2/feedforward_layer_norm/gamma_11transformer_layer_2/feedforward_layer_norm/beta_17transformer_layer_3/self_attention_layer/query/kernel_15transformer_layer_3/self_attention_layer/query/bias_15transformer_layer_3/self_attention_layer/key/kernel_13transformer_layer_3/self_attention_layer/key/bias_17transformer_layer_3/self_attention_layer/value/kernel_15transformer_layer_3/self_attention_layer/value/bias_1Btransformer_layer_3/self_attention_layer/attention_output/kernel_1@transformer_layer_3/self_attention_layer/attention_output/bias_15transformer_layer_3/self_attention_layer_norm/gamma_14transformer_layer_3/self_attention_layer_norm/beta_1;transformer_layer_3/feedforward_intermediate_dense/kernel_19transformer_layer_3/feedforward_intermediate_dense/bias_15transformer_layer_3/feedforward_output_dense/kernel_13transformer_layer_3/feedforward_output_dense/bias_12transformer_layer_3/feedforward_layer_norm/gamma_11transformer_layer_3/feedforward_layer_norm/beta_17transformer_layer_4/self_attention_layer/query/kernel_15transformer_layer_4/self_attention_layer/query/bias_15transformer_layer_4/self_attention_layer/key/kernel_13transformer_layer_4/self_attention_layer/key/bias_17transformer_layer_4/self_attention_layer/value/kernel_15transformer_layer_4/self_attention_layer/value/bias_1Btransformer_layer_4/self_attention_layer/attention_output/kernel_1@transformer_layer_4/self_attention_layer/attention_output/bias_15transformer_layer_4/self_attention_layer_norm/gamma_14transformer_layer_4/self_attention_layer_norm/beta_1;transformer_layer_4/feedforward_intermediate_dense/kernel_19transformer_layer_4/feedforward_intermediate_dense/bias_15transformer_layer_4/feedforward_output_dense/kernel_13transformer_layer_4/feedforward_output_dense/bias_12transformer_layer_4/feedforward_layer_norm/gamma_11transformer_layer_4/feedforward_layer_norm/beta_17transformer_layer_5/self_attention_layer/query/kernel_15transformer_layer_5/self_attention_layer/query/bias_15transformer_layer_5/self_attention_layer/key/kernel_13transformer_layer_5/self_attention_layer/key/bias_17transformer_layer_5/self_attention_layer/value/kernel_15transformer_layer_5/self_attention_layer/value/bias_1Btransformer_layer_5/self_attention_layer/attention_output/kernel_1@transformer_layer_5/self_attention_layer/attention_output/bias_15transformer_layer_5/self_attention_layer_norm/gamma_14transformer_layer_5/self_attention_layer_norm/beta_1;transformer_layer_5/feedforward_intermediate_dense/kernel_19transformer_layer_5/feedforward_intermediate_dense/bias_15transformer_layer_5/feedforward_output_dense/kernel_13transformer_layer_5/feedforward_output_dense/bias_12transformer_layer_5/feedforward_layer_norm/gamma_11transformer_layer_5/feedforward_layer_norm/beta_1pooled_dense/kernel_1pooled_dense/bias_1logits/kernel_1logits/bias_1*u
Tinn
l2j*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*�
_read_only_resource_inputsl
jh	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghi*-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_signature_wrapper___call___8525

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures*
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30
'31
(32
)33
*34
+35
,36
-37
.38
/39
040
141
242
343
444
545
646
747
848
949
:50
;51
<52
=53
>54
?55
@56
A57
B58
C59
D60
E61
F62
G63
H64
I65
J66
K67
L68
M69
N70
O71
P72
Q73
R74
S75
T76
U77
V78
W79
X80
Y81
Z82
[83
\84
]85
^86
_87
`88
a89
b90
c91
d92
e93
f94
g95
h96
i97
j98
k99
l100
m101
n102
o103
p104
q105
r106
s107
t108
u109
v110
w111
x112
y113
z114
{115
|116
}117
~118
119
�120
�121
�122
�123*
#
0
�1
�2
�3*
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30
'31
(32
)33
*34
+35
,36
-37
.38
/39
040
141
242
343
444
545
646
747
848
949
:50
;51
<52
=53
>54
?55
@56
A57
B58
C59
D60
E61
F62
G63
H64
I65
J66
K67
L68
M69
N70
O71
P72
Q73
R74
S75
T76
U77
V78
W79
X80
Y81
Z82
[83
\84
]85
^86
_87
`88
a89
b90
c91
d92
e93
f94
g95
h96
i97
j98
k99
l100
m101
n102
o103
p104
q105
r106
s107
t108
u109
v110
w111
x112
y113
z114
{115
|116
}117
~118
�119*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83
�84
�85
�86
�87
�88
�89
�90
�91
�92
�93
�94
�95
�96
�97
�98
�99
�100
�101
�102
�103*
* 

�trace_0* 
$

�serve
�serving_default* 
LF
VARIABLE_VALUEVariable_123&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_122&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_121&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_120&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_119&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_118&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_117&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_116&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_115&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_114&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_113'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_112'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_111'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_110'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_109'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_108'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_107'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_106'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_105'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_104'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_103'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_102'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_101'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_100'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_99'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_98'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_97'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_96'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_95'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_94'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_93'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_92'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_91'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_90'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_89'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_88'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_87'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_86'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_85'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_84'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_83'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_82'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_81'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_80'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_79'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_78'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_77'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_76'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_75'variables/48/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_74'variables/49/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_73'variables/50/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_72'variables/51/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_71'variables/52/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_70'variables/53/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_69'variables/54/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_68'variables/55/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_67'variables/56/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_66'variables/57/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_65'variables/58/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_64'variables/59/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_63'variables/60/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_62'variables/61/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_61'variables/62/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_60'variables/63/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_59'variables/64/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_58'variables/65/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_57'variables/66/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_56'variables/67/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_55'variables/68/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_54'variables/69/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_53'variables/70/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_52'variables/71/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_51'variables/72/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_50'variables/73/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_49'variables/74/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_48'variables/75/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_47'variables/76/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_46'variables/77/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_45'variables/78/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_44'variables/79/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_43'variables/80/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_42'variables/81/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_41'variables/82/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_40'variables/83/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_39'variables/84/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_38'variables/85/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_37'variables/86/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_36'variables/87/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_35'variables/88/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_34'variables/89/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_33'variables/90/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_32'variables/91/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_31'variables/92/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_30'variables/93/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_29'variables/94/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_28'variables/95/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_27'variables/96/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_26'variables/97/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_25'variables/98/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_24'variables/99/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_23(variables/100/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_22(variables/101/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_21(variables/102/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_20(variables/103/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_19(variables/104/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_18(variables/105/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_17(variables/106/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_16(variables/107/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_15(variables/108/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_14(variables/109/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_13(variables/110/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_12(variables/111/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_11(variables/112/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEVariable_10(variables/113/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE
Variable_9(variables/114/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE
Variable_8(variables/115/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE
Variable_7(variables/116/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE
Variable_6(variables/117/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE
Variable_5(variables/118/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE
Variable_4(variables/119/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE
Variable_3(variables/120/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE
Variable_2(variables/121/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE
Variable_1(variables/122/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUEVariable(variables/123/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE7transformer_layer_3/self_attention_layer/query/kernel_1+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE5transformer_layer_3/self_attention_layer/value/bias_1+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE@transformer_layer_3/self_attention_layer/attention_output/bias_1+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE3transformer_layer_3/feedforward_output_dense/bias_1+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE4transformer_layer_5/self_attention_layer_norm/beta_1+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE5transformer_layer_1/self_attention_layer/query/bias_1+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE5transformer_layer_3/self_attention_layer/key/kernel_1+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE4transformer_layer_3/self_attention_layer_norm/beta_1+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE3transformer_layer_1/self_attention_layer/key/bias_1+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE3transformer_layer_4/feedforward_output_dense/bias_1+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE2transformer_layer_4/feedforward_layer_norm/gamma_1,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE1transformer_layer_0/feedforward_layer_norm/beta_1,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_4/self_attention_layer/query/bias_1,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_4/self_attention_layer/key/kernel_1,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE9token_and_position_embedding/token_embedding/embeddings_1,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE3transformer_layer_5/feedforward_output_dense/bias_1,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_3/self_attention_layer/query/bias_1,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE3transformer_layer_3/self_attention_layer/key/bias_1,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_0/self_attention_layer_norm/gamma_1,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE1transformer_layer_4/feedforward_layer_norm/beta_1,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEembeddings_layer_norm/beta_1,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE7transformer_layer_2/self_attention_layer/query/kernel_1,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE2transformer_layer_2/feedforward_layer_norm/gamma_1,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE9transformer_layer_4/feedforward_intermediate_dense/bias_1,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE3transformer_layer_4/self_attention_layer/key/bias_1,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE7transformer_layer_4/self_attention_layer/value/kernel_1,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_1/feedforward_output_dense/kernel_1,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEBtransformer_layer_0/self_attention_layer/attention_output/kernel_1,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE2transformer_layer_1/feedforward_layer_norm/gamma_1,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE9transformer_layer_1/feedforward_intermediate_dense/bias_1,_all_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEpooled_dense/bias_1,_all_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_2/self_attention_layer/query/bias_1,_all_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE1transformer_layer_2/feedforward_layer_norm/beta_1,_all_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE;transformer_layer_4/feedforward_intermediate_dense/kernel_1,_all_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEBtransformer_layer_4/self_attention_layer/attention_output/kernel_1,_all_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE<token_and_position_embedding/position_embedding/embeddings_1,_all_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_2/self_attention_layer/key/kernel_1,_all_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE9transformer_layer_2/feedforward_intermediate_dense/bias_1,_all_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_4/self_attention_layer/value/bias_1,_all_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE2transformer_layer_5/feedforward_layer_norm/gamma_1,_all_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUElogits/kernel_1,_all_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEembeddings_layer_norm/gamma_1,_all_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE7transformer_layer_2/self_attention_layer/value/kernel_1,_all_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE2transformer_layer_3/feedforward_layer_norm/gamma_1,_all_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_4/self_attention_layer_norm/gamma_1,_all_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE7transformer_layer_5/self_attention_layer/query/kernel_1,_all_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE7transformer_layer_0/self_attention_layer/query/kernel_1,_all_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE@transformer_layer_0/self_attention_layer/attention_output/bias_1,_all_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE1transformer_layer_1/feedforward_layer_norm/beta_1,_all_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_5/self_attention_layer/key/kernel_1,_all_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_0/self_attention_layer/key/kernel_1,_all_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE4transformer_layer_0/self_attention_layer_norm/beta_1,_all_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE7transformer_layer_5/self_attention_layer/value/kernel_1,_all_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE7transformer_layer_1/self_attention_layer/value/kernel_1,_all_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE3transformer_layer_2/self_attention_layer/key/bias_1,_all_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE;transformer_layer_2/feedforward_intermediate_dense/kernel_1,_all_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE@transformer_layer_4/self_attention_layer/attention_output/bias_1,_all_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE1transformer_layer_5/feedforward_layer_norm/beta_1,_all_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEpooled_dense/kernel_1,_all_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUElogits/bias_1,_all_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE3transformer_layer_0/feedforward_output_dense/bias_1,_all_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_2/self_attention_layer/value/bias_1,_all_variables/61/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEBtransformer_layer_2/self_attention_layer/attention_output/kernel_1,_all_variables/62/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE1transformer_layer_3/feedforward_layer_norm/beta_1,_all_variables/63/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE4transformer_layer_4/self_attention_layer_norm/beta_1,_all_variables/64/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_5/self_attention_layer/query/bias_1,_all_variables/65/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE;transformer_layer_5/feedforward_intermediate_dense/kernel_1,_all_variables/66/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_0/self_attention_layer/query/bias_1,_all_variables/67/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_2/self_attention_layer_norm/gamma_1,_all_variables/68/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE;transformer_layer_3/feedforward_intermediate_dense/kernel_1,_all_variables/69/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE3transformer_layer_0/self_attention_layer/key/bias_1,_all_variables/70/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE;transformer_layer_1/feedforward_intermediate_dense/kernel_1,_all_variables/71/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_5/self_attention_layer/value/bias_1,_all_variables/72/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEBtransformer_layer_5/self_attention_layer/attention_output/kernel_1,_all_variables/73/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE7transformer_layer_0/self_attention_layer/value/kernel_1,_all_variables/74/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_1/self_attention_layer/value/bias_1,_all_variables/75/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEBtransformer_layer_1/self_attention_layer/attention_output/kernel_1,_all_variables/76/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE;transformer_layer_0/feedforward_intermediate_dense/kernel_1,_all_variables/77/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_1/self_attention_layer_norm/gamma_1,_all_variables/78/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE@transformer_layer_2/self_attention_layer/attention_output/bias_1,_all_variables/79/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_2/feedforward_output_dense/kernel_1,_all_variables/80/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE9transformer_layer_5/feedforward_intermediate_dense/bias_1,_all_variables/81/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE4transformer_layer_2/self_attention_layer_norm/beta_1,_all_variables/82/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE9transformer_layer_3/feedforward_intermediate_dense/bias_1,_all_variables/83/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE3transformer_layer_5/self_attention_layer/key/bias_1,_all_variables/84/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE3transformer_layer_1/feedforward_output_dense/bias_1,_all_variables/85/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE7transformer_layer_3/self_attention_layer/value/kernel_1,_all_variables/86/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEBtransformer_layer_3/self_attention_layer/attention_output/kernel_1,_all_variables/87/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_3/feedforward_output_dense/kernel_1,_all_variables/88/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE@transformer_layer_5/self_attention_layer/attention_output/bias_1,_all_variables/89/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_5/self_attention_layer_norm/gamma_1,_all_variables/90/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_0/self_attention_layer/value/bias_1,_all_variables/91/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE7transformer_layer_1/self_attention_layer/query/kernel_1,_all_variables/92/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE@transformer_layer_1/self_attention_layer/attention_output/bias_1,_all_variables/93/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_3/self_attention_layer_norm/gamma_1,_all_variables/94/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE9transformer_layer_0/feedforward_intermediate_dense/bias_1,_all_variables/95/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_1/self_attention_layer/key/kernel_1,_all_variables/96/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE4transformer_layer_1/self_attention_layer_norm/beta_1,_all_variables/97/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE5transformer_layer_4/feedforward_output_dense/kernel_1,_all_variables/98/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE7transformer_layer_4/self_attention_layer/query/kernel_1,_all_variables/99/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE5transformer_layer_0/feedforward_output_dense/kernel_1-_all_variables/100/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE2transformer_layer_0/feedforward_layer_norm/gamma_1-_all_variables/101/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE3transformer_layer_2/feedforward_output_dense/bias_1-_all_variables/102/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE5transformer_layer_5/feedforward_output_dense/kernel_1-_all_variables/103/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�=
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_123Variable_122Variable_121Variable_120Variable_119Variable_118Variable_117Variable_116Variable_115Variable_114Variable_113Variable_112Variable_111Variable_110Variable_109Variable_108Variable_107Variable_106Variable_105Variable_104Variable_103Variable_102Variable_101Variable_100Variable_99Variable_98Variable_97Variable_96Variable_95Variable_94Variable_93Variable_92Variable_91Variable_90Variable_89Variable_88Variable_87Variable_86Variable_85Variable_84Variable_83Variable_82Variable_81Variable_80Variable_79Variable_78Variable_77Variable_76Variable_75Variable_74Variable_73Variable_72Variable_71Variable_70Variable_69Variable_68Variable_67Variable_66Variable_65Variable_64Variable_63Variable_62Variable_61Variable_60Variable_59Variable_58Variable_57Variable_56Variable_55Variable_54Variable_53Variable_52Variable_51Variable_50Variable_49Variable_48Variable_47Variable_46Variable_45Variable_44Variable_43Variable_42Variable_41Variable_40Variable_39Variable_38Variable_37Variable_36Variable_35Variable_34Variable_33Variable_32Variable_31Variable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable7transformer_layer_3/self_attention_layer/query/kernel_15transformer_layer_3/self_attention_layer/value/bias_1@transformer_layer_3/self_attention_layer/attention_output/bias_13transformer_layer_3/feedforward_output_dense/bias_14transformer_layer_5/self_attention_layer_norm/beta_15transformer_layer_1/self_attention_layer/query/bias_15transformer_layer_3/self_attention_layer/key/kernel_14transformer_layer_3/self_attention_layer_norm/beta_13transformer_layer_1/self_attention_layer/key/bias_13transformer_layer_4/feedforward_output_dense/bias_12transformer_layer_4/feedforward_layer_norm/gamma_11transformer_layer_0/feedforward_layer_norm/beta_15transformer_layer_4/self_attention_layer/query/bias_15transformer_layer_4/self_attention_layer/key/kernel_19token_and_position_embedding/token_embedding/embeddings_13transformer_layer_5/feedforward_output_dense/bias_15transformer_layer_3/self_attention_layer/query/bias_13transformer_layer_3/self_attention_layer/key/bias_15transformer_layer_0/self_attention_layer_norm/gamma_11transformer_layer_4/feedforward_layer_norm/beta_1embeddings_layer_norm/beta_17transformer_layer_2/self_attention_layer/query/kernel_12transformer_layer_2/feedforward_layer_norm/gamma_19transformer_layer_4/feedforward_intermediate_dense/bias_13transformer_layer_4/self_attention_layer/key/bias_17transformer_layer_4/self_attention_layer/value/kernel_15transformer_layer_1/feedforward_output_dense/kernel_1Btransformer_layer_0/self_attention_layer/attention_output/kernel_12transformer_layer_1/feedforward_layer_norm/gamma_19transformer_layer_1/feedforward_intermediate_dense/bias_1pooled_dense/bias_15transformer_layer_2/self_attention_layer/query/bias_11transformer_layer_2/feedforward_layer_norm/beta_1;transformer_layer_4/feedforward_intermediate_dense/kernel_1Btransformer_layer_4/self_attention_layer/attention_output/kernel_1<token_and_position_embedding/position_embedding/embeddings_15transformer_layer_2/self_attention_layer/key/kernel_19transformer_layer_2/feedforward_intermediate_dense/bias_15transformer_layer_4/self_attention_layer/value/bias_12transformer_layer_5/feedforward_layer_norm/gamma_1logits/kernel_1embeddings_layer_norm/gamma_17transformer_layer_2/self_attention_layer/value/kernel_12transformer_layer_3/feedforward_layer_norm/gamma_15transformer_layer_4/self_attention_layer_norm/gamma_17transformer_layer_5/self_attention_layer/query/kernel_17transformer_layer_0/self_attention_layer/query/kernel_1@transformer_layer_0/self_attention_layer/attention_output/bias_11transformer_layer_1/feedforward_layer_norm/beta_15transformer_layer_5/self_attention_layer/key/kernel_15transformer_layer_0/self_attention_layer/key/kernel_14transformer_layer_0/self_attention_layer_norm/beta_17transformer_layer_5/self_attention_layer/value/kernel_17transformer_layer_1/self_attention_layer/value/kernel_13transformer_layer_2/self_attention_layer/key/bias_1;transformer_layer_2/feedforward_intermediate_dense/kernel_1@transformer_layer_4/self_attention_layer/attention_output/bias_11transformer_layer_5/feedforward_layer_norm/beta_1pooled_dense/kernel_1logits/bias_13transformer_layer_0/feedforward_output_dense/bias_15transformer_layer_2/self_attention_layer/value/bias_1Btransformer_layer_2/self_attention_layer/attention_output/kernel_11transformer_layer_3/feedforward_layer_norm/beta_14transformer_layer_4/self_attention_layer_norm/beta_15transformer_layer_5/self_attention_layer/query/bias_1;transformer_layer_5/feedforward_intermediate_dense/kernel_15transformer_layer_0/self_attention_layer/query/bias_15transformer_layer_2/self_attention_layer_norm/gamma_1;transformer_layer_3/feedforward_intermediate_dense/kernel_13transformer_layer_0/self_attention_layer/key/bias_1;transformer_layer_1/feedforward_intermediate_dense/kernel_15transformer_layer_5/self_attention_layer/value/bias_1Btransformer_layer_5/self_attention_layer/attention_output/kernel_17transformer_layer_0/self_attention_layer/value/kernel_15transformer_layer_1/self_attention_layer/value/bias_1Btransformer_layer_1/self_attention_layer/attention_output/kernel_1;transformer_layer_0/feedforward_intermediate_dense/kernel_15transformer_layer_1/self_attention_layer_norm/gamma_1@transformer_layer_2/self_attention_layer/attention_output/bias_15transformer_layer_2/feedforward_output_dense/kernel_19transformer_layer_5/feedforward_intermediate_dense/bias_14transformer_layer_2/self_attention_layer_norm/beta_19transformer_layer_3/feedforward_intermediate_dense/bias_13transformer_layer_5/self_attention_layer/key/bias_13transformer_layer_1/feedforward_output_dense/bias_17transformer_layer_3/self_attention_layer/value/kernel_1Btransformer_layer_3/self_attention_layer/attention_output/kernel_15transformer_layer_3/feedforward_output_dense/kernel_1@transformer_layer_5/self_attention_layer/attention_output/bias_15transformer_layer_5/self_attention_layer_norm/gamma_15transformer_layer_0/self_attention_layer/value/bias_17transformer_layer_1/self_attention_layer/query/kernel_1@transformer_layer_1/self_attention_layer/attention_output/bias_15transformer_layer_3/self_attention_layer_norm/gamma_19transformer_layer_0/feedforward_intermediate_dense/bias_15transformer_layer_1/self_attention_layer/key/kernel_14transformer_layer_1/self_attention_layer_norm/beta_15transformer_layer_4/feedforward_output_dense/kernel_17transformer_layer_4/self_attention_layer/query/kernel_15transformer_layer_0/feedforward_output_dense/kernel_12transformer_layer_0/feedforward_layer_norm/gamma_13transformer_layer_2/feedforward_output_dense/bias_15transformer_layer_5/feedforward_output_dense/kernel_1Const*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_10415
�=
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable_123Variable_122Variable_121Variable_120Variable_119Variable_118Variable_117Variable_116Variable_115Variable_114Variable_113Variable_112Variable_111Variable_110Variable_109Variable_108Variable_107Variable_106Variable_105Variable_104Variable_103Variable_102Variable_101Variable_100Variable_99Variable_98Variable_97Variable_96Variable_95Variable_94Variable_93Variable_92Variable_91Variable_90Variable_89Variable_88Variable_87Variable_86Variable_85Variable_84Variable_83Variable_82Variable_81Variable_80Variable_79Variable_78Variable_77Variable_76Variable_75Variable_74Variable_73Variable_72Variable_71Variable_70Variable_69Variable_68Variable_67Variable_66Variable_65Variable_64Variable_63Variable_62Variable_61Variable_60Variable_59Variable_58Variable_57Variable_56Variable_55Variable_54Variable_53Variable_52Variable_51Variable_50Variable_49Variable_48Variable_47Variable_46Variable_45Variable_44Variable_43Variable_42Variable_41Variable_40Variable_39Variable_38Variable_37Variable_36Variable_35Variable_34Variable_33Variable_32Variable_31Variable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable7transformer_layer_3/self_attention_layer/query/kernel_15transformer_layer_3/self_attention_layer/value/bias_1@transformer_layer_3/self_attention_layer/attention_output/bias_13transformer_layer_3/feedforward_output_dense/bias_14transformer_layer_5/self_attention_layer_norm/beta_15transformer_layer_1/self_attention_layer/query/bias_15transformer_layer_3/self_attention_layer/key/kernel_14transformer_layer_3/self_attention_layer_norm/beta_13transformer_layer_1/self_attention_layer/key/bias_13transformer_layer_4/feedforward_output_dense/bias_12transformer_layer_4/feedforward_layer_norm/gamma_11transformer_layer_0/feedforward_layer_norm/beta_15transformer_layer_4/self_attention_layer/query/bias_15transformer_layer_4/self_attention_layer/key/kernel_19token_and_position_embedding/token_embedding/embeddings_13transformer_layer_5/feedforward_output_dense/bias_15transformer_layer_3/self_attention_layer/query/bias_13transformer_layer_3/self_attention_layer/key/bias_15transformer_layer_0/self_attention_layer_norm/gamma_11transformer_layer_4/feedforward_layer_norm/beta_1embeddings_layer_norm/beta_17transformer_layer_2/self_attention_layer/query/kernel_12transformer_layer_2/feedforward_layer_norm/gamma_19transformer_layer_4/feedforward_intermediate_dense/bias_13transformer_layer_4/self_attention_layer/key/bias_17transformer_layer_4/self_attention_layer/value/kernel_15transformer_layer_1/feedforward_output_dense/kernel_1Btransformer_layer_0/self_attention_layer/attention_output/kernel_12transformer_layer_1/feedforward_layer_norm/gamma_19transformer_layer_1/feedforward_intermediate_dense/bias_1pooled_dense/bias_15transformer_layer_2/self_attention_layer/query/bias_11transformer_layer_2/feedforward_layer_norm/beta_1;transformer_layer_4/feedforward_intermediate_dense/kernel_1Btransformer_layer_4/self_attention_layer/attention_output/kernel_1<token_and_position_embedding/position_embedding/embeddings_15transformer_layer_2/self_attention_layer/key/kernel_19transformer_layer_2/feedforward_intermediate_dense/bias_15transformer_layer_4/self_attention_layer/value/bias_12transformer_layer_5/feedforward_layer_norm/gamma_1logits/kernel_1embeddings_layer_norm/gamma_17transformer_layer_2/self_attention_layer/value/kernel_12transformer_layer_3/feedforward_layer_norm/gamma_15transformer_layer_4/self_attention_layer_norm/gamma_17transformer_layer_5/self_attention_layer/query/kernel_17transformer_layer_0/self_attention_layer/query/kernel_1@transformer_layer_0/self_attention_layer/attention_output/bias_11transformer_layer_1/feedforward_layer_norm/beta_15transformer_layer_5/self_attention_layer/key/kernel_15transformer_layer_0/self_attention_layer/key/kernel_14transformer_layer_0/self_attention_layer_norm/beta_17transformer_layer_5/self_attention_layer/value/kernel_17transformer_layer_1/self_attention_layer/value/kernel_13transformer_layer_2/self_attention_layer/key/bias_1;transformer_layer_2/feedforward_intermediate_dense/kernel_1@transformer_layer_4/self_attention_layer/attention_output/bias_11transformer_layer_5/feedforward_layer_norm/beta_1pooled_dense/kernel_1logits/bias_13transformer_layer_0/feedforward_output_dense/bias_15transformer_layer_2/self_attention_layer/value/bias_1Btransformer_layer_2/self_attention_layer/attention_output/kernel_11transformer_layer_3/feedforward_layer_norm/beta_14transformer_layer_4/self_attention_layer_norm/beta_15transformer_layer_5/self_attention_layer/query/bias_1;transformer_layer_5/feedforward_intermediate_dense/kernel_15transformer_layer_0/self_attention_layer/query/bias_15transformer_layer_2/self_attention_layer_norm/gamma_1;transformer_layer_3/feedforward_intermediate_dense/kernel_13transformer_layer_0/self_attention_layer/key/bias_1;transformer_layer_1/feedforward_intermediate_dense/kernel_15transformer_layer_5/self_attention_layer/value/bias_1Btransformer_layer_5/self_attention_layer/attention_output/kernel_17transformer_layer_0/self_attention_layer/value/kernel_15transformer_layer_1/self_attention_layer/value/bias_1Btransformer_layer_1/self_attention_layer/attention_output/kernel_1;transformer_layer_0/feedforward_intermediate_dense/kernel_15transformer_layer_1/self_attention_layer_norm/gamma_1@transformer_layer_2/self_attention_layer/attention_output/bias_15transformer_layer_2/feedforward_output_dense/kernel_19transformer_layer_5/feedforward_intermediate_dense/bias_14transformer_layer_2/self_attention_layer_norm/beta_19transformer_layer_3/feedforward_intermediate_dense/bias_13transformer_layer_5/self_attention_layer/key/bias_13transformer_layer_1/feedforward_output_dense/bias_17transformer_layer_3/self_attention_layer/value/kernel_1Btransformer_layer_3/self_attention_layer/attention_output/kernel_15transformer_layer_3/feedforward_output_dense/kernel_1@transformer_layer_5/self_attention_layer/attention_output/bias_15transformer_layer_5/self_attention_layer_norm/gamma_15transformer_layer_0/self_attention_layer/value/bias_17transformer_layer_1/self_attention_layer/query/kernel_1@transformer_layer_1/self_attention_layer/attention_output/bias_15transformer_layer_3/self_attention_layer_norm/gamma_19transformer_layer_0/feedforward_intermediate_dense/bias_15transformer_layer_1/self_attention_layer/key/kernel_14transformer_layer_1/self_attention_layer_norm/beta_15transformer_layer_4/feedforward_output_dense/kernel_17transformer_layer_4/self_attention_layer/query/kernel_15transformer_layer_0/feedforward_output_dense/kernel_12transformer_layer_0/feedforward_layer_norm/gamma_13transformer_layer_2/feedforward_output_dense/bias_15transformer_layer_5/feedforward_output_dense/kernel_1*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_11108��(
��
��
!__inference__traced_restore_11108
file_prefix2
assignvariableop_variable_123:���3
assignvariableop_1_variable_122:
��.
assignvariableop_2_variable_121:	�.
assignvariableop_3_variable_120:	�-
assignvariableop_4_variable_119:6
assignvariableop_5_variable_118:�@1
assignvariableop_6_variable_117:@6
assignvariableop_7_variable_116:�@1
assignvariableop_8_variable_115:@6
assignvariableop_9_variable_114:�@2
 assignvariableop_10_variable_113:@.
 assignvariableop_11_variable_112:7
 assignvariableop_12_variable_111:@�/
 assignvariableop_13_variable_110:	�/
 assignvariableop_14_variable_109:	�/
 assignvariableop_15_variable_108:	�.
 assignvariableop_16_variable_107:/
 assignvariableop_17_variable_106:	�/
 assignvariableop_18_variable_105:	�4
 assignvariableop_19_variable_104:
��/
 assignvariableop_20_variable_103:	�4
 assignvariableop_21_variable_102:
��/
 assignvariableop_22_variable_101:	�.
 assignvariableop_23_variable_100:6
assignvariableop_24_variable_99:�@1
assignvariableop_25_variable_98:@6
assignvariableop_26_variable_97:�@1
assignvariableop_27_variable_96:@6
assignvariableop_28_variable_95:�@1
assignvariableop_29_variable_94:@-
assignvariableop_30_variable_93:6
assignvariableop_31_variable_92:@�.
assignvariableop_32_variable_91:	�.
assignvariableop_33_variable_90:	�.
assignvariableop_34_variable_89:	�-
assignvariableop_35_variable_88:.
assignvariableop_36_variable_87:	�.
assignvariableop_37_variable_86:	�3
assignvariableop_38_variable_85:
��.
assignvariableop_39_variable_84:	�3
assignvariableop_40_variable_83:
��.
assignvariableop_41_variable_82:	�-
assignvariableop_42_variable_81:6
assignvariableop_43_variable_80:�@1
assignvariableop_44_variable_79:@6
assignvariableop_45_variable_78:�@1
assignvariableop_46_variable_77:@6
assignvariableop_47_variable_76:�@1
assignvariableop_48_variable_75:@-
assignvariableop_49_variable_74:6
assignvariableop_50_variable_73:@�.
assignvariableop_51_variable_72:	�.
assignvariableop_52_variable_71:	�.
assignvariableop_53_variable_70:	�-
assignvariableop_54_variable_69:.
assignvariableop_55_variable_68:	�.
assignvariableop_56_variable_67:	�3
assignvariableop_57_variable_66:
��.
assignvariableop_58_variable_65:	�3
assignvariableop_59_variable_64:
��.
assignvariableop_60_variable_63:	�-
assignvariableop_61_variable_62:6
assignvariableop_62_variable_61:�@1
assignvariableop_63_variable_60:@6
assignvariableop_64_variable_59:�@1
assignvariableop_65_variable_58:@6
assignvariableop_66_variable_57:�@1
assignvariableop_67_variable_56:@-
assignvariableop_68_variable_55:6
assignvariableop_69_variable_54:@�.
assignvariableop_70_variable_53:	�.
assignvariableop_71_variable_52:	�.
assignvariableop_72_variable_51:	�-
assignvariableop_73_variable_50:.
assignvariableop_74_variable_49:	�.
assignvariableop_75_variable_48:	�3
assignvariableop_76_variable_47:
��.
assignvariableop_77_variable_46:	�3
assignvariableop_78_variable_45:
��.
assignvariableop_79_variable_44:	�-
assignvariableop_80_variable_43:6
assignvariableop_81_variable_42:�@1
assignvariableop_82_variable_41:@6
assignvariableop_83_variable_40:�@1
assignvariableop_84_variable_39:@6
assignvariableop_85_variable_38:�@1
assignvariableop_86_variable_37:@-
assignvariableop_87_variable_36:6
assignvariableop_88_variable_35:@�.
assignvariableop_89_variable_34:	�.
assignvariableop_90_variable_33:	�.
assignvariableop_91_variable_32:	�-
assignvariableop_92_variable_31:.
assignvariableop_93_variable_30:	�.
assignvariableop_94_variable_29:	�3
assignvariableop_95_variable_28:
��.
assignvariableop_96_variable_27:	�3
assignvariableop_97_variable_26:
��.
assignvariableop_98_variable_25:	�-
assignvariableop_99_variable_24:7
 assignvariableop_100_variable_23:�@2
 assignvariableop_101_variable_22:@7
 assignvariableop_102_variable_21:�@2
 assignvariableop_103_variable_20:@7
 assignvariableop_104_variable_19:�@2
 assignvariableop_105_variable_18:@.
 assignvariableop_106_variable_17:7
 assignvariableop_107_variable_16:@�/
 assignvariableop_108_variable_15:	�/
 assignvariableop_109_variable_14:	�/
 assignvariableop_110_variable_13:	�.
 assignvariableop_111_variable_12:/
 assignvariableop_112_variable_11:	�/
 assignvariableop_113_variable_10:	�3
assignvariableop_114_variable_9:
��.
assignvariableop_115_variable_8:	�3
assignvariableop_116_variable_7:
��.
assignvariableop_117_variable_6:	�-
assignvariableop_118_variable_5:3
assignvariableop_119_variable_4:
��.
assignvariableop_120_variable_3:	�-
assignvariableop_121_variable_2:2
assignvariableop_122_variable_1:	�+
assignvariableop_123_variable:c
Lassignvariableop_124_transformer_layer_3_self_attention_layer_query_kernel_1:�@\
Jassignvariableop_125_transformer_layer_3_self_attention_layer_value_bias_1:@d
Uassignvariableop_126_transformer_layer_3_self_attention_layer_attention_output_bias_1:	�W
Hassignvariableop_127_transformer_layer_3_feedforward_output_dense_bias_1:	�X
Iassignvariableop_128_transformer_layer_5_self_attention_layer_norm_beta_1:	�\
Jassignvariableop_129_transformer_layer_1_self_attention_layer_query_bias_1:@a
Jassignvariableop_130_transformer_layer_3_self_attention_layer_key_kernel_1:�@X
Iassignvariableop_131_transformer_layer_3_self_attention_layer_norm_beta_1:	�Z
Hassignvariableop_132_transformer_layer_1_self_attention_layer_key_bias_1:@W
Hassignvariableop_133_transformer_layer_4_feedforward_output_dense_bias_1:	�V
Gassignvariableop_134_transformer_layer_4_feedforward_layer_norm_gamma_1:	�U
Fassignvariableop_135_transformer_layer_0_feedforward_layer_norm_beta_1:	�\
Jassignvariableop_136_transformer_layer_4_self_attention_layer_query_bias_1:@a
Jassignvariableop_137_transformer_layer_4_self_attention_layer_key_kernel_1:�@c
Nassignvariableop_138_token_and_position_embedding_token_embedding_embeddings_1:���W
Hassignvariableop_139_transformer_layer_5_feedforward_output_dense_bias_1:	�\
Jassignvariableop_140_transformer_layer_3_self_attention_layer_query_bias_1:@Z
Hassignvariableop_141_transformer_layer_3_self_attention_layer_key_bias_1:@Y
Jassignvariableop_142_transformer_layer_0_self_attention_layer_norm_gamma_1:	�U
Fassignvariableop_143_transformer_layer_4_feedforward_layer_norm_beta_1:	�@
1assignvariableop_144_embeddings_layer_norm_beta_1:	�c
Lassignvariableop_145_transformer_layer_2_self_attention_layer_query_kernel_1:�@V
Gassignvariableop_146_transformer_layer_2_feedforward_layer_norm_gamma_1:	�]
Nassignvariableop_147_transformer_layer_4_feedforward_intermediate_dense_bias_1:	�Z
Hassignvariableop_148_transformer_layer_4_self_attention_layer_key_bias_1:@c
Lassignvariableop_149_transformer_layer_4_self_attention_layer_value_kernel_1:�@^
Jassignvariableop_150_transformer_layer_1_feedforward_output_dense_kernel_1:
��n
Wassignvariableop_151_transformer_layer_0_self_attention_layer_attention_output_kernel_1:@�V
Gassignvariableop_152_transformer_layer_1_feedforward_layer_norm_gamma_1:	�]
Nassignvariableop_153_transformer_layer_1_feedforward_intermediate_dense_bias_1:	�7
(assignvariableop_154_pooled_dense_bias_1:	�\
Jassignvariableop_155_transformer_layer_2_self_attention_layer_query_bias_1:@U
Fassignvariableop_156_transformer_layer_2_feedforward_layer_norm_beta_1:	�d
Passignvariableop_157_transformer_layer_4_feedforward_intermediate_dense_kernel_1:
��n
Wassignvariableop_158_transformer_layer_4_self_attention_layer_attention_output_kernel_1:@�e
Qassignvariableop_159_token_and_position_embedding_position_embedding_embeddings_1:
��a
Jassignvariableop_160_transformer_layer_2_self_attention_layer_key_kernel_1:�@]
Nassignvariableop_161_transformer_layer_2_feedforward_intermediate_dense_bias_1:	�\
Jassignvariableop_162_transformer_layer_4_self_attention_layer_value_bias_1:@V
Gassignvariableop_163_transformer_layer_5_feedforward_layer_norm_gamma_1:	�7
$assignvariableop_164_logits_kernel_1:	�A
2assignvariableop_165_embeddings_layer_norm_gamma_1:	�c
Lassignvariableop_166_transformer_layer_2_self_attention_layer_value_kernel_1:�@V
Gassignvariableop_167_transformer_layer_3_feedforward_layer_norm_gamma_1:	�Y
Jassignvariableop_168_transformer_layer_4_self_attention_layer_norm_gamma_1:	�c
Lassignvariableop_169_transformer_layer_5_self_attention_layer_query_kernel_1:�@c
Lassignvariableop_170_transformer_layer_0_self_attention_layer_query_kernel_1:�@d
Uassignvariableop_171_transformer_layer_0_self_attention_layer_attention_output_bias_1:	�U
Fassignvariableop_172_transformer_layer_1_feedforward_layer_norm_beta_1:	�a
Jassignvariableop_173_transformer_layer_5_self_attention_layer_key_kernel_1:�@a
Jassignvariableop_174_transformer_layer_0_self_attention_layer_key_kernel_1:�@X
Iassignvariableop_175_transformer_layer_0_self_attention_layer_norm_beta_1:	�c
Lassignvariableop_176_transformer_layer_5_self_attention_layer_value_kernel_1:�@c
Lassignvariableop_177_transformer_layer_1_self_attention_layer_value_kernel_1:�@Z
Hassignvariableop_178_transformer_layer_2_self_attention_layer_key_bias_1:@d
Passignvariableop_179_transformer_layer_2_feedforward_intermediate_dense_kernel_1:
��d
Uassignvariableop_180_transformer_layer_4_self_attention_layer_attention_output_bias_1:	�U
Fassignvariableop_181_transformer_layer_5_feedforward_layer_norm_beta_1:	�>
*assignvariableop_182_pooled_dense_kernel_1:
��0
"assignvariableop_183_logits_bias_1:W
Hassignvariableop_184_transformer_layer_0_feedforward_output_dense_bias_1:	�\
Jassignvariableop_185_transformer_layer_2_self_attention_layer_value_bias_1:@n
Wassignvariableop_186_transformer_layer_2_self_attention_layer_attention_output_kernel_1:@�U
Fassignvariableop_187_transformer_layer_3_feedforward_layer_norm_beta_1:	�X
Iassignvariableop_188_transformer_layer_4_self_attention_layer_norm_beta_1:	�\
Jassignvariableop_189_transformer_layer_5_self_attention_layer_query_bias_1:@d
Passignvariableop_190_transformer_layer_5_feedforward_intermediate_dense_kernel_1:
��\
Jassignvariableop_191_transformer_layer_0_self_attention_layer_query_bias_1:@Y
Jassignvariableop_192_transformer_layer_2_self_attention_layer_norm_gamma_1:	�d
Passignvariableop_193_transformer_layer_3_feedforward_intermediate_dense_kernel_1:
��Z
Hassignvariableop_194_transformer_layer_0_self_attention_layer_key_bias_1:@d
Passignvariableop_195_transformer_layer_1_feedforward_intermediate_dense_kernel_1:
��\
Jassignvariableop_196_transformer_layer_5_self_attention_layer_value_bias_1:@n
Wassignvariableop_197_transformer_layer_5_self_attention_layer_attention_output_kernel_1:@�c
Lassignvariableop_198_transformer_layer_0_self_attention_layer_value_kernel_1:�@\
Jassignvariableop_199_transformer_layer_1_self_attention_layer_value_bias_1:@n
Wassignvariableop_200_transformer_layer_1_self_attention_layer_attention_output_kernel_1:@�d
Passignvariableop_201_transformer_layer_0_feedforward_intermediate_dense_kernel_1:
��Y
Jassignvariableop_202_transformer_layer_1_self_attention_layer_norm_gamma_1:	�d
Uassignvariableop_203_transformer_layer_2_self_attention_layer_attention_output_bias_1:	�^
Jassignvariableop_204_transformer_layer_2_feedforward_output_dense_kernel_1:
��]
Nassignvariableop_205_transformer_layer_5_feedforward_intermediate_dense_bias_1:	�X
Iassignvariableop_206_transformer_layer_2_self_attention_layer_norm_beta_1:	�]
Nassignvariableop_207_transformer_layer_3_feedforward_intermediate_dense_bias_1:	�Z
Hassignvariableop_208_transformer_layer_5_self_attention_layer_key_bias_1:@W
Hassignvariableop_209_transformer_layer_1_feedforward_output_dense_bias_1:	�c
Lassignvariableop_210_transformer_layer_3_self_attention_layer_value_kernel_1:�@n
Wassignvariableop_211_transformer_layer_3_self_attention_layer_attention_output_kernel_1:@�^
Jassignvariableop_212_transformer_layer_3_feedforward_output_dense_kernel_1:
��d
Uassignvariableop_213_transformer_layer_5_self_attention_layer_attention_output_bias_1:	�Y
Jassignvariableop_214_transformer_layer_5_self_attention_layer_norm_gamma_1:	�\
Jassignvariableop_215_transformer_layer_0_self_attention_layer_value_bias_1:@c
Lassignvariableop_216_transformer_layer_1_self_attention_layer_query_kernel_1:�@d
Uassignvariableop_217_transformer_layer_1_self_attention_layer_attention_output_bias_1:	�Y
Jassignvariableop_218_transformer_layer_3_self_attention_layer_norm_gamma_1:	�]
Nassignvariableop_219_transformer_layer_0_feedforward_intermediate_dense_bias_1:	�a
Jassignvariableop_220_transformer_layer_1_self_attention_layer_key_kernel_1:�@X
Iassignvariableop_221_transformer_layer_1_self_attention_layer_norm_beta_1:	�^
Jassignvariableop_222_transformer_layer_4_feedforward_output_dense_kernel_1:
��c
Lassignvariableop_223_transformer_layer_4_self_attention_layer_query_kernel_1:�@^
Jassignvariableop_224_transformer_layer_0_feedforward_output_dense_kernel_1:
��V
Gassignvariableop_225_transformer_layer_0_feedforward_layer_norm_gamma_1:	�W
Hassignvariableop_226_transformer_layer_2_feedforward_output_dense_bias_1:	�^
Jassignvariableop_227_transformer_layer_5_feedforward_output_dense_kernel_1:
��
identity_229��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_136�AssignVariableOp_137�AssignVariableOp_138�AssignVariableOp_139�AssignVariableOp_14�AssignVariableOp_140�AssignVariableOp_141�AssignVariableOp_142�AssignVariableOp_143�AssignVariableOp_144�AssignVariableOp_145�AssignVariableOp_146�AssignVariableOp_147�AssignVariableOp_148�AssignVariableOp_149�AssignVariableOp_15�AssignVariableOp_150�AssignVariableOp_151�AssignVariableOp_152�AssignVariableOp_153�AssignVariableOp_154�AssignVariableOp_155�AssignVariableOp_156�AssignVariableOp_157�AssignVariableOp_158�AssignVariableOp_159�AssignVariableOp_16�AssignVariableOp_160�AssignVariableOp_161�AssignVariableOp_162�AssignVariableOp_163�AssignVariableOp_164�AssignVariableOp_165�AssignVariableOp_166�AssignVariableOp_167�AssignVariableOp_168�AssignVariableOp_169�AssignVariableOp_17�AssignVariableOp_170�AssignVariableOp_171�AssignVariableOp_172�AssignVariableOp_173�AssignVariableOp_174�AssignVariableOp_175�AssignVariableOp_176�AssignVariableOp_177�AssignVariableOp_178�AssignVariableOp_179�AssignVariableOp_18�AssignVariableOp_180�AssignVariableOp_181�AssignVariableOp_182�AssignVariableOp_183�AssignVariableOp_184�AssignVariableOp_185�AssignVariableOp_186�AssignVariableOp_187�AssignVariableOp_188�AssignVariableOp_189�AssignVariableOp_19�AssignVariableOp_190�AssignVariableOp_191�AssignVariableOp_192�AssignVariableOp_193�AssignVariableOp_194�AssignVariableOp_195�AssignVariableOp_196�AssignVariableOp_197�AssignVariableOp_198�AssignVariableOp_199�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_200�AssignVariableOp_201�AssignVariableOp_202�AssignVariableOp_203�AssignVariableOp_204�AssignVariableOp_205�AssignVariableOp_206�AssignVariableOp_207�AssignVariableOp_208�AssignVariableOp_209�AssignVariableOp_21�AssignVariableOp_210�AssignVariableOp_211�AssignVariableOp_212�AssignVariableOp_213�AssignVariableOp_214�AssignVariableOp_215�AssignVariableOp_216�AssignVariableOp_217�AssignVariableOp_218�AssignVariableOp_219�AssignVariableOp_22�AssignVariableOp_220�AssignVariableOp_221�AssignVariableOp_222�AssignVariableOp_223�AssignVariableOp_224�AssignVariableOp_225�AssignVariableOp_226�AssignVariableOp_227�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�N
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�M
value�MB�M�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB'variables/70/.ATTRIBUTES/VARIABLE_VALUEB'variables/71/.ATTRIBUTES/VARIABLE_VALUEB'variables/72/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB'variables/75/.ATTRIBUTES/VARIABLE_VALUEB'variables/76/.ATTRIBUTES/VARIABLE_VALUEB'variables/77/.ATTRIBUTES/VARIABLE_VALUEB'variables/78/.ATTRIBUTES/VARIABLE_VALUEB'variables/79/.ATTRIBUTES/VARIABLE_VALUEB'variables/80/.ATTRIBUTES/VARIABLE_VALUEB'variables/81/.ATTRIBUTES/VARIABLE_VALUEB'variables/82/.ATTRIBUTES/VARIABLE_VALUEB'variables/83/.ATTRIBUTES/VARIABLE_VALUEB'variables/84/.ATTRIBUTES/VARIABLE_VALUEB'variables/85/.ATTRIBUTES/VARIABLE_VALUEB'variables/86/.ATTRIBUTES/VARIABLE_VALUEB'variables/87/.ATTRIBUTES/VARIABLE_VALUEB'variables/88/.ATTRIBUTES/VARIABLE_VALUEB'variables/89/.ATTRIBUTES/VARIABLE_VALUEB'variables/90/.ATTRIBUTES/VARIABLE_VALUEB'variables/91/.ATTRIBUTES/VARIABLE_VALUEB'variables/92/.ATTRIBUTES/VARIABLE_VALUEB'variables/93/.ATTRIBUTES/VARIABLE_VALUEB'variables/94/.ATTRIBUTES/VARIABLE_VALUEB'variables/95/.ATTRIBUTES/VARIABLE_VALUEB'variables/96/.ATTRIBUTES/VARIABLE_VALUEB'variables/97/.ATTRIBUTES/VARIABLE_VALUEB'variables/98/.ATTRIBUTES/VARIABLE_VALUEB'variables/99/.ATTRIBUTES/VARIABLE_VALUEB(variables/100/.ATTRIBUTES/VARIABLE_VALUEB(variables/101/.ATTRIBUTES/VARIABLE_VALUEB(variables/102/.ATTRIBUTES/VARIABLE_VALUEB(variables/103/.ATTRIBUTES/VARIABLE_VALUEB(variables/104/.ATTRIBUTES/VARIABLE_VALUEB(variables/105/.ATTRIBUTES/VARIABLE_VALUEB(variables/106/.ATTRIBUTES/VARIABLE_VALUEB(variables/107/.ATTRIBUTES/VARIABLE_VALUEB(variables/108/.ATTRIBUTES/VARIABLE_VALUEB(variables/109/.ATTRIBUTES/VARIABLE_VALUEB(variables/110/.ATTRIBUTES/VARIABLE_VALUEB(variables/111/.ATTRIBUTES/VARIABLE_VALUEB(variables/112/.ATTRIBUTES/VARIABLE_VALUEB(variables/113/.ATTRIBUTES/VARIABLE_VALUEB(variables/114/.ATTRIBUTES/VARIABLE_VALUEB(variables/115/.ATTRIBUTES/VARIABLE_VALUEB(variables/116/.ATTRIBUTES/VARIABLE_VALUEB(variables/117/.ATTRIBUTES/VARIABLE_VALUEB(variables/118/.ATTRIBUTES/VARIABLE_VALUEB(variables/119/.ATTRIBUTES/VARIABLE_VALUEB(variables/120/.ATTRIBUTES/VARIABLE_VALUEB(variables/121/.ATTRIBUTES/VARIABLE_VALUEB(variables/122/.ATTRIBUTES/VARIABLE_VALUEB(variables/123/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/29/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/30/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/31/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/32/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/33/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/34/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/35/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/36/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/37/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/38/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/39/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/40/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/41/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/42/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/43/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/44/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/45/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/46/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/47/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/48/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/49/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/50/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/51/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/52/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/53/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/54/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/55/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/56/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/57/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/58/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/59/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/60/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/61/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/62/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/63/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/64/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/65/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/66/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/67/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/68/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/69/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/70/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/71/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/72/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/73/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/74/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/75/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/76/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/77/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/78/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/79/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/80/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/81/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/82/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/83/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/84/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/85/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/86/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/87/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/88/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/89/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/90/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/91/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/92/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/93/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/94/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/95/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/96/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/97/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/98/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/99/.ATTRIBUTES/VARIABLE_VALUEB-_all_variables/100/.ATTRIBUTES/VARIABLE_VALUEB-_all_variables/101/.ATTRIBUTES/VARIABLE_VALUEB-_all_variables/102/.ATTRIBUTES/VARIABLE_VALUEB-_all_variables/103/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �

	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_123Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_122Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_121Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_120Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_119Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_118Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_117Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_116Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_115Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_114Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp assignvariableop_10_variable_113Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp assignvariableop_11_variable_112Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp assignvariableop_12_variable_111Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_variable_110Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp assignvariableop_14_variable_109Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp assignvariableop_15_variable_108Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp assignvariableop_16_variable_107Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp assignvariableop_17_variable_106Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp assignvariableop_18_variable_105Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp assignvariableop_19_variable_104Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp assignvariableop_20_variable_103Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp assignvariableop_21_variable_102Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp assignvariableop_22_variable_101Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp assignvariableop_23_variable_100Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variable_99Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_variable_98Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_variable_97Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_variable_96Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_variable_95Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_variable_94Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_variable_93Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_variable_92Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_variable_91Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_variable_90Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_variable_89Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_variable_88Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_variable_87Identity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_variable_86Identity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_variable_85Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_variable_84Identity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpassignvariableop_40_variable_83Identity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_variable_82Identity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpassignvariableop_42_variable_81Identity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpassignvariableop_43_variable_80Identity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpassignvariableop_44_variable_79Identity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpassignvariableop_45_variable_78Identity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpassignvariableop_46_variable_77Identity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpassignvariableop_47_variable_76Identity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpassignvariableop_48_variable_75Identity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpassignvariableop_49_variable_74Identity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_variable_73Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpassignvariableop_51_variable_72Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpassignvariableop_52_variable_71Identity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpassignvariableop_53_variable_70Identity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpassignvariableop_54_variable_69Identity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpassignvariableop_55_variable_68Identity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpassignvariableop_56_variable_67Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpassignvariableop_57_variable_66Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpassignvariableop_58_variable_65Identity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpassignvariableop_59_variable_64Identity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpassignvariableop_60_variable_63Identity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpassignvariableop_61_variable_62Identity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpassignvariableop_62_variable_61Identity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpassignvariableop_63_variable_60Identity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOpassignvariableop_64_variable_59Identity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOpassignvariableop_65_variable_58Identity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOpassignvariableop_66_variable_57Identity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOpassignvariableop_67_variable_56Identity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpassignvariableop_68_variable_55Identity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOpassignvariableop_69_variable_54Identity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOpassignvariableop_70_variable_53Identity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOpassignvariableop_71_variable_52Identity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOpassignvariableop_72_variable_51Identity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOpassignvariableop_73_variable_50Identity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOpassignvariableop_74_variable_49Identity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOpassignvariableop_75_variable_48Identity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOpassignvariableop_76_variable_47Identity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOpassignvariableop_77_variable_46Identity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOpassignvariableop_78_variable_45Identity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOpassignvariableop_79_variable_44Identity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOpassignvariableop_80_variable_43Identity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOpassignvariableop_81_variable_42Identity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOpassignvariableop_82_variable_41Identity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOpassignvariableop_83_variable_40Identity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOpassignvariableop_84_variable_39Identity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOpassignvariableop_85_variable_38Identity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOpassignvariableop_86_variable_37Identity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOpassignvariableop_87_variable_36Identity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOpassignvariableop_88_variable_35Identity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOpassignvariableop_89_variable_34Identity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOpassignvariableop_90_variable_33Identity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOpassignvariableop_91_variable_32Identity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOpassignvariableop_92_variable_31Identity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOpassignvariableop_93_variable_30Identity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOpassignvariableop_94_variable_29Identity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOpassignvariableop_95_variable_28Identity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOpassignvariableop_96_variable_27Identity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOpassignvariableop_97_variable_26Identity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOpassignvariableop_98_variable_25Identity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOpassignvariableop_99_variable_24Identity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp assignvariableop_100_variable_23Identity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp assignvariableop_101_variable_22Identity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp assignvariableop_102_variable_21Identity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp assignvariableop_103_variable_20Identity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp assignvariableop_104_variable_19Identity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp assignvariableop_105_variable_18Identity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp assignvariableop_106_variable_17Identity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp assignvariableop_107_variable_16Identity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp assignvariableop_108_variable_15Identity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp assignvariableop_109_variable_14Identity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp assignvariableop_110_variable_13Identity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp assignvariableop_111_variable_12Identity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp assignvariableop_112_variable_11Identity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp assignvariableop_113_variable_10Identity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOpassignvariableop_114_variable_9Identity_114:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOpassignvariableop_115_variable_8Identity_115:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOpassignvariableop_116_variable_7Identity_116:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOpassignvariableop_117_variable_6Identity_117:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOpassignvariableop_118_variable_5Identity_118:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOpassignvariableop_119_variable_4Identity_119:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOpassignvariableop_120_variable_3Identity_120:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOpassignvariableop_121_variable_2Identity_121:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_122AssignVariableOpassignvariableop_122_variable_1Identity_122:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_123AssignVariableOpassignvariableop_123_variableIdentity_123:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_124AssignVariableOpLassignvariableop_124_transformer_layer_3_self_attention_layer_query_kernel_1Identity_124:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_125AssignVariableOpJassignvariableop_125_transformer_layer_3_self_attention_layer_value_bias_1Identity_125:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_126AssignVariableOpUassignvariableop_126_transformer_layer_3_self_attention_layer_attention_output_bias_1Identity_126:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_127AssignVariableOpHassignvariableop_127_transformer_layer_3_feedforward_output_dense_bias_1Identity_127:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_128AssignVariableOpIassignvariableop_128_transformer_layer_5_self_attention_layer_norm_beta_1Identity_128:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_129AssignVariableOpJassignvariableop_129_transformer_layer_1_self_attention_layer_query_bias_1Identity_129:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_130AssignVariableOpJassignvariableop_130_transformer_layer_3_self_attention_layer_key_kernel_1Identity_130:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_131AssignVariableOpIassignvariableop_131_transformer_layer_3_self_attention_layer_norm_beta_1Identity_131:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_132AssignVariableOpHassignvariableop_132_transformer_layer_1_self_attention_layer_key_bias_1Identity_132:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_133AssignVariableOpHassignvariableop_133_transformer_layer_4_feedforward_output_dense_bias_1Identity_133:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_134AssignVariableOpGassignvariableop_134_transformer_layer_4_feedforward_layer_norm_gamma_1Identity_134:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_135AssignVariableOpFassignvariableop_135_transformer_layer_0_feedforward_layer_norm_beta_1Identity_135:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_136AssignVariableOpJassignvariableop_136_transformer_layer_4_self_attention_layer_query_bias_1Identity_136:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_137AssignVariableOpJassignvariableop_137_transformer_layer_4_self_attention_layer_key_kernel_1Identity_137:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_138AssignVariableOpNassignvariableop_138_token_and_position_embedding_token_embedding_embeddings_1Identity_138:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_139AssignVariableOpHassignvariableop_139_transformer_layer_5_feedforward_output_dense_bias_1Identity_139:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_140AssignVariableOpJassignvariableop_140_transformer_layer_3_self_attention_layer_query_bias_1Identity_140:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_141AssignVariableOpHassignvariableop_141_transformer_layer_3_self_attention_layer_key_bias_1Identity_141:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_142AssignVariableOpJassignvariableop_142_transformer_layer_0_self_attention_layer_norm_gamma_1Identity_142:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_143AssignVariableOpFassignvariableop_143_transformer_layer_4_feedforward_layer_norm_beta_1Identity_143:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_144AssignVariableOp1assignvariableop_144_embeddings_layer_norm_beta_1Identity_144:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_145AssignVariableOpLassignvariableop_145_transformer_layer_2_self_attention_layer_query_kernel_1Identity_145:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_146AssignVariableOpGassignvariableop_146_transformer_layer_2_feedforward_layer_norm_gamma_1Identity_146:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_147AssignVariableOpNassignvariableop_147_transformer_layer_4_feedforward_intermediate_dense_bias_1Identity_147:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_148AssignVariableOpHassignvariableop_148_transformer_layer_4_self_attention_layer_key_bias_1Identity_148:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_149AssignVariableOpLassignvariableop_149_transformer_layer_4_self_attention_layer_value_kernel_1Identity_149:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_150AssignVariableOpJassignvariableop_150_transformer_layer_1_feedforward_output_dense_kernel_1Identity_150:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_151AssignVariableOpWassignvariableop_151_transformer_layer_0_self_attention_layer_attention_output_kernel_1Identity_151:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_152AssignVariableOpGassignvariableop_152_transformer_layer_1_feedforward_layer_norm_gamma_1Identity_152:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_153AssignVariableOpNassignvariableop_153_transformer_layer_1_feedforward_intermediate_dense_bias_1Identity_153:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_154AssignVariableOp(assignvariableop_154_pooled_dense_bias_1Identity_154:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_155AssignVariableOpJassignvariableop_155_transformer_layer_2_self_attention_layer_query_bias_1Identity_155:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_156AssignVariableOpFassignvariableop_156_transformer_layer_2_feedforward_layer_norm_beta_1Identity_156:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_157AssignVariableOpPassignvariableop_157_transformer_layer_4_feedforward_intermediate_dense_kernel_1Identity_157:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_158AssignVariableOpWassignvariableop_158_transformer_layer_4_self_attention_layer_attention_output_kernel_1Identity_158:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_159AssignVariableOpQassignvariableop_159_token_and_position_embedding_position_embedding_embeddings_1Identity_159:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_160AssignVariableOpJassignvariableop_160_transformer_layer_2_self_attention_layer_key_kernel_1Identity_160:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_161AssignVariableOpNassignvariableop_161_transformer_layer_2_feedforward_intermediate_dense_bias_1Identity_161:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_162AssignVariableOpJassignvariableop_162_transformer_layer_4_self_attention_layer_value_bias_1Identity_162:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_163AssignVariableOpGassignvariableop_163_transformer_layer_5_feedforward_layer_norm_gamma_1Identity_163:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_164AssignVariableOp$assignvariableop_164_logits_kernel_1Identity_164:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_165IdentityRestoreV2:tensors:165"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_165AssignVariableOp2assignvariableop_165_embeddings_layer_norm_gamma_1Identity_165:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_166IdentityRestoreV2:tensors:166"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_166AssignVariableOpLassignvariableop_166_transformer_layer_2_self_attention_layer_value_kernel_1Identity_166:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_167IdentityRestoreV2:tensors:167"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_167AssignVariableOpGassignvariableop_167_transformer_layer_3_feedforward_layer_norm_gamma_1Identity_167:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_168IdentityRestoreV2:tensors:168"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_168AssignVariableOpJassignvariableop_168_transformer_layer_4_self_attention_layer_norm_gamma_1Identity_168:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_169IdentityRestoreV2:tensors:169"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_169AssignVariableOpLassignvariableop_169_transformer_layer_5_self_attention_layer_query_kernel_1Identity_169:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_170IdentityRestoreV2:tensors:170"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_170AssignVariableOpLassignvariableop_170_transformer_layer_0_self_attention_layer_query_kernel_1Identity_170:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_171IdentityRestoreV2:tensors:171"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_171AssignVariableOpUassignvariableop_171_transformer_layer_0_self_attention_layer_attention_output_bias_1Identity_171:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_172IdentityRestoreV2:tensors:172"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_172AssignVariableOpFassignvariableop_172_transformer_layer_1_feedforward_layer_norm_beta_1Identity_172:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_173IdentityRestoreV2:tensors:173"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_173AssignVariableOpJassignvariableop_173_transformer_layer_5_self_attention_layer_key_kernel_1Identity_173:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_174IdentityRestoreV2:tensors:174"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_174AssignVariableOpJassignvariableop_174_transformer_layer_0_self_attention_layer_key_kernel_1Identity_174:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_175IdentityRestoreV2:tensors:175"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_175AssignVariableOpIassignvariableop_175_transformer_layer_0_self_attention_layer_norm_beta_1Identity_175:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_176IdentityRestoreV2:tensors:176"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_176AssignVariableOpLassignvariableop_176_transformer_layer_5_self_attention_layer_value_kernel_1Identity_176:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_177IdentityRestoreV2:tensors:177"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_177AssignVariableOpLassignvariableop_177_transformer_layer_1_self_attention_layer_value_kernel_1Identity_177:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_178IdentityRestoreV2:tensors:178"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_178AssignVariableOpHassignvariableop_178_transformer_layer_2_self_attention_layer_key_bias_1Identity_178:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_179IdentityRestoreV2:tensors:179"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_179AssignVariableOpPassignvariableop_179_transformer_layer_2_feedforward_intermediate_dense_kernel_1Identity_179:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_180IdentityRestoreV2:tensors:180"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_180AssignVariableOpUassignvariableop_180_transformer_layer_4_self_attention_layer_attention_output_bias_1Identity_180:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_181IdentityRestoreV2:tensors:181"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_181AssignVariableOpFassignvariableop_181_transformer_layer_5_feedforward_layer_norm_beta_1Identity_181:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_182IdentityRestoreV2:tensors:182"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_182AssignVariableOp*assignvariableop_182_pooled_dense_kernel_1Identity_182:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_183IdentityRestoreV2:tensors:183"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_183AssignVariableOp"assignvariableop_183_logits_bias_1Identity_183:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_184IdentityRestoreV2:tensors:184"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_184AssignVariableOpHassignvariableop_184_transformer_layer_0_feedforward_output_dense_bias_1Identity_184:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_185IdentityRestoreV2:tensors:185"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_185AssignVariableOpJassignvariableop_185_transformer_layer_2_self_attention_layer_value_bias_1Identity_185:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_186IdentityRestoreV2:tensors:186"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_186AssignVariableOpWassignvariableop_186_transformer_layer_2_self_attention_layer_attention_output_kernel_1Identity_186:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_187IdentityRestoreV2:tensors:187"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_187AssignVariableOpFassignvariableop_187_transformer_layer_3_feedforward_layer_norm_beta_1Identity_187:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_188IdentityRestoreV2:tensors:188"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_188AssignVariableOpIassignvariableop_188_transformer_layer_4_self_attention_layer_norm_beta_1Identity_188:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_189IdentityRestoreV2:tensors:189"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_189AssignVariableOpJassignvariableop_189_transformer_layer_5_self_attention_layer_query_bias_1Identity_189:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_190IdentityRestoreV2:tensors:190"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_190AssignVariableOpPassignvariableop_190_transformer_layer_5_feedforward_intermediate_dense_kernel_1Identity_190:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_191IdentityRestoreV2:tensors:191"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_191AssignVariableOpJassignvariableop_191_transformer_layer_0_self_attention_layer_query_bias_1Identity_191:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_192IdentityRestoreV2:tensors:192"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_192AssignVariableOpJassignvariableop_192_transformer_layer_2_self_attention_layer_norm_gamma_1Identity_192:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_193IdentityRestoreV2:tensors:193"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_193AssignVariableOpPassignvariableop_193_transformer_layer_3_feedforward_intermediate_dense_kernel_1Identity_193:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_194IdentityRestoreV2:tensors:194"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_194AssignVariableOpHassignvariableop_194_transformer_layer_0_self_attention_layer_key_bias_1Identity_194:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_195IdentityRestoreV2:tensors:195"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_195AssignVariableOpPassignvariableop_195_transformer_layer_1_feedforward_intermediate_dense_kernel_1Identity_195:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_196IdentityRestoreV2:tensors:196"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_196AssignVariableOpJassignvariableop_196_transformer_layer_5_self_attention_layer_value_bias_1Identity_196:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_197IdentityRestoreV2:tensors:197"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_197AssignVariableOpWassignvariableop_197_transformer_layer_5_self_attention_layer_attention_output_kernel_1Identity_197:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_198IdentityRestoreV2:tensors:198"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_198AssignVariableOpLassignvariableop_198_transformer_layer_0_self_attention_layer_value_kernel_1Identity_198:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_199IdentityRestoreV2:tensors:199"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_199AssignVariableOpJassignvariableop_199_transformer_layer_1_self_attention_layer_value_bias_1Identity_199:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_200IdentityRestoreV2:tensors:200"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_200AssignVariableOpWassignvariableop_200_transformer_layer_1_self_attention_layer_attention_output_kernel_1Identity_200:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_201IdentityRestoreV2:tensors:201"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_201AssignVariableOpPassignvariableop_201_transformer_layer_0_feedforward_intermediate_dense_kernel_1Identity_201:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_202IdentityRestoreV2:tensors:202"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_202AssignVariableOpJassignvariableop_202_transformer_layer_1_self_attention_layer_norm_gamma_1Identity_202:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_203IdentityRestoreV2:tensors:203"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_203AssignVariableOpUassignvariableop_203_transformer_layer_2_self_attention_layer_attention_output_bias_1Identity_203:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_204IdentityRestoreV2:tensors:204"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_204AssignVariableOpJassignvariableop_204_transformer_layer_2_feedforward_output_dense_kernel_1Identity_204:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_205IdentityRestoreV2:tensors:205"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_205AssignVariableOpNassignvariableop_205_transformer_layer_5_feedforward_intermediate_dense_bias_1Identity_205:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_206IdentityRestoreV2:tensors:206"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_206AssignVariableOpIassignvariableop_206_transformer_layer_2_self_attention_layer_norm_beta_1Identity_206:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_207IdentityRestoreV2:tensors:207"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_207AssignVariableOpNassignvariableop_207_transformer_layer_3_feedforward_intermediate_dense_bias_1Identity_207:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_208IdentityRestoreV2:tensors:208"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_208AssignVariableOpHassignvariableop_208_transformer_layer_5_self_attention_layer_key_bias_1Identity_208:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_209IdentityRestoreV2:tensors:209"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_209AssignVariableOpHassignvariableop_209_transformer_layer_1_feedforward_output_dense_bias_1Identity_209:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_210IdentityRestoreV2:tensors:210"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_210AssignVariableOpLassignvariableop_210_transformer_layer_3_self_attention_layer_value_kernel_1Identity_210:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_211IdentityRestoreV2:tensors:211"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_211AssignVariableOpWassignvariableop_211_transformer_layer_3_self_attention_layer_attention_output_kernel_1Identity_211:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_212IdentityRestoreV2:tensors:212"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_212AssignVariableOpJassignvariableop_212_transformer_layer_3_feedforward_output_dense_kernel_1Identity_212:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_213IdentityRestoreV2:tensors:213"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_213AssignVariableOpUassignvariableop_213_transformer_layer_5_self_attention_layer_attention_output_bias_1Identity_213:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_214IdentityRestoreV2:tensors:214"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_214AssignVariableOpJassignvariableop_214_transformer_layer_5_self_attention_layer_norm_gamma_1Identity_214:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_215IdentityRestoreV2:tensors:215"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_215AssignVariableOpJassignvariableop_215_transformer_layer_0_self_attention_layer_value_bias_1Identity_215:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_216IdentityRestoreV2:tensors:216"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_216AssignVariableOpLassignvariableop_216_transformer_layer_1_self_attention_layer_query_kernel_1Identity_216:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_217IdentityRestoreV2:tensors:217"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_217AssignVariableOpUassignvariableop_217_transformer_layer_1_self_attention_layer_attention_output_bias_1Identity_217:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_218IdentityRestoreV2:tensors:218"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_218AssignVariableOpJassignvariableop_218_transformer_layer_3_self_attention_layer_norm_gamma_1Identity_218:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_219IdentityRestoreV2:tensors:219"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_219AssignVariableOpNassignvariableop_219_transformer_layer_0_feedforward_intermediate_dense_bias_1Identity_219:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_220IdentityRestoreV2:tensors:220"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_220AssignVariableOpJassignvariableop_220_transformer_layer_1_self_attention_layer_key_kernel_1Identity_220:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_221IdentityRestoreV2:tensors:221"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_221AssignVariableOpIassignvariableop_221_transformer_layer_1_self_attention_layer_norm_beta_1Identity_221:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_222IdentityRestoreV2:tensors:222"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_222AssignVariableOpJassignvariableop_222_transformer_layer_4_feedforward_output_dense_kernel_1Identity_222:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_223IdentityRestoreV2:tensors:223"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_223AssignVariableOpLassignvariableop_223_transformer_layer_4_self_attention_layer_query_kernel_1Identity_223:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_224IdentityRestoreV2:tensors:224"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_224AssignVariableOpJassignvariableop_224_transformer_layer_0_feedforward_output_dense_kernel_1Identity_224:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_225IdentityRestoreV2:tensors:225"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_225AssignVariableOpGassignvariableop_225_transformer_layer_0_feedforward_layer_norm_gamma_1Identity_225:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_226IdentityRestoreV2:tensors:226"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_226AssignVariableOpHassignvariableop_226_transformer_layer_2_feedforward_output_dense_bias_1Identity_226:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_227IdentityRestoreV2:tensors:227"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_227AssignVariableOpJassignvariableop_227_transformer_layer_5_feedforward_output_dense_kernel_1Identity_227:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �(
Identity_228Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_181^AssignVariableOp_182^AssignVariableOp_183^AssignVariableOp_184^AssignVariableOp_185^AssignVariableOp_186^AssignVariableOp_187^AssignVariableOp_188^AssignVariableOp_189^AssignVariableOp_19^AssignVariableOp_190^AssignVariableOp_191^AssignVariableOp_192^AssignVariableOp_193^AssignVariableOp_194^AssignVariableOp_195^AssignVariableOp_196^AssignVariableOp_197^AssignVariableOp_198^AssignVariableOp_199^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_200^AssignVariableOp_201^AssignVariableOp_202^AssignVariableOp_203^AssignVariableOp_204^AssignVariableOp_205^AssignVariableOp_206^AssignVariableOp_207^AssignVariableOp_208^AssignVariableOp_209^AssignVariableOp_21^AssignVariableOp_210^AssignVariableOp_211^AssignVariableOp_212^AssignVariableOp_213^AssignVariableOp_214^AssignVariableOp_215^AssignVariableOp_216^AssignVariableOp_217^AssignVariableOp_218^AssignVariableOp_219^AssignVariableOp_22^AssignVariableOp_220^AssignVariableOp_221^AssignVariableOp_222^AssignVariableOp_223^AssignVariableOp_224^AssignVariableOp_225^AssignVariableOp_226^AssignVariableOp_227^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_229IdentityIdentity_228:output:0^NoOp_1*
T0*
_output_shapes
: �(
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_165^AssignVariableOp_166^AssignVariableOp_167^AssignVariableOp_168^AssignVariableOp_169^AssignVariableOp_17^AssignVariableOp_170^AssignVariableOp_171^AssignVariableOp_172^AssignVariableOp_173^AssignVariableOp_174^AssignVariableOp_175^AssignVariableOp_176^AssignVariableOp_177^AssignVariableOp_178^AssignVariableOp_179^AssignVariableOp_18^AssignVariableOp_180^AssignVariableOp_181^AssignVariableOp_182^AssignVariableOp_183^AssignVariableOp_184^AssignVariableOp_185^AssignVariableOp_186^AssignVariableOp_187^AssignVariableOp_188^AssignVariableOp_189^AssignVariableOp_19^AssignVariableOp_190^AssignVariableOp_191^AssignVariableOp_192^AssignVariableOp_193^AssignVariableOp_194^AssignVariableOp_195^AssignVariableOp_196^AssignVariableOp_197^AssignVariableOp_198^AssignVariableOp_199^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_200^AssignVariableOp_201^AssignVariableOp_202^AssignVariableOp_203^AssignVariableOp_204^AssignVariableOp_205^AssignVariableOp_206^AssignVariableOp_207^AssignVariableOp_208^AssignVariableOp_209^AssignVariableOp_21^AssignVariableOp_210^AssignVariableOp_211^AssignVariableOp_212^AssignVariableOp_213^AssignVariableOp_214^AssignVariableOp_215^AssignVariableOp_216^AssignVariableOp_217^AssignVariableOp_218^AssignVariableOp_219^AssignVariableOp_22^AssignVariableOp_220^AssignVariableOp_221^AssignVariableOp_222^AssignVariableOp_223^AssignVariableOp_224^AssignVariableOp_225^AssignVariableOp_226^AssignVariableOp_227^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_229Identity_229:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642,
AssignVariableOp_165AssignVariableOp_1652,
AssignVariableOp_166AssignVariableOp_1662,
AssignVariableOp_167AssignVariableOp_1672,
AssignVariableOp_168AssignVariableOp_1682,
AssignVariableOp_169AssignVariableOp_1692*
AssignVariableOp_17AssignVariableOp_172,
AssignVariableOp_170AssignVariableOp_1702,
AssignVariableOp_171AssignVariableOp_1712,
AssignVariableOp_172AssignVariableOp_1722,
AssignVariableOp_173AssignVariableOp_1732,
AssignVariableOp_174AssignVariableOp_1742,
AssignVariableOp_175AssignVariableOp_1752,
AssignVariableOp_176AssignVariableOp_1762,
AssignVariableOp_177AssignVariableOp_1772,
AssignVariableOp_178AssignVariableOp_1782,
AssignVariableOp_179AssignVariableOp_1792*
AssignVariableOp_18AssignVariableOp_182,
AssignVariableOp_180AssignVariableOp_1802,
AssignVariableOp_181AssignVariableOp_1812,
AssignVariableOp_182AssignVariableOp_1822,
AssignVariableOp_183AssignVariableOp_1832,
AssignVariableOp_184AssignVariableOp_1842,
AssignVariableOp_185AssignVariableOp_1852,
AssignVariableOp_186AssignVariableOp_1862,
AssignVariableOp_187AssignVariableOp_1872,
AssignVariableOp_188AssignVariableOp_1882,
AssignVariableOp_189AssignVariableOp_1892*
AssignVariableOp_19AssignVariableOp_192,
AssignVariableOp_190AssignVariableOp_1902,
AssignVariableOp_191AssignVariableOp_1912,
AssignVariableOp_192AssignVariableOp_1922,
AssignVariableOp_193AssignVariableOp_1932,
AssignVariableOp_194AssignVariableOp_1942,
AssignVariableOp_195AssignVariableOp_1952,
AssignVariableOp_196AssignVariableOp_1962,
AssignVariableOp_197AssignVariableOp_1972,
AssignVariableOp_198AssignVariableOp_1982,
AssignVariableOp_199AssignVariableOp_1992(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202,
AssignVariableOp_200AssignVariableOp_2002,
AssignVariableOp_201AssignVariableOp_2012,
AssignVariableOp_202AssignVariableOp_2022,
AssignVariableOp_203AssignVariableOp_2032,
AssignVariableOp_204AssignVariableOp_2042,
AssignVariableOp_205AssignVariableOp_2052,
AssignVariableOp_206AssignVariableOp_2062,
AssignVariableOp_207AssignVariableOp_2072,
AssignVariableOp_208AssignVariableOp_2082,
AssignVariableOp_209AssignVariableOp_2092*
AssignVariableOp_21AssignVariableOp_212,
AssignVariableOp_210AssignVariableOp_2102,
AssignVariableOp_211AssignVariableOp_2112,
AssignVariableOp_212AssignVariableOp_2122,
AssignVariableOp_213AssignVariableOp_2132,
AssignVariableOp_214AssignVariableOp_2142,
AssignVariableOp_215AssignVariableOp_2152,
AssignVariableOp_216AssignVariableOp_2162,
AssignVariableOp_217AssignVariableOp_2172,
AssignVariableOp_218AssignVariableOp_2182,
AssignVariableOp_219AssignVariableOp_2192*
AssignVariableOp_22AssignVariableOp_222,
AssignVariableOp_220AssignVariableOp_2202,
AssignVariableOp_221AssignVariableOp_2212,
AssignVariableOp_222AssignVariableOp_2222,
AssignVariableOp_223AssignVariableOp_2232,
AssignVariableOp_224AssignVariableOp_2242,
AssignVariableOp_225AssignVariableOp_2252,
AssignVariableOp_226AssignVariableOp_2262,
AssignVariableOp_227AssignVariableOp_2272*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_user_specified_nameVariable_123:,(
&
_user_specified_nameVariable_122:,(
&
_user_specified_nameVariable_121:,(
&
_user_specified_nameVariable_120:,(
&
_user_specified_nameVariable_119:,(
&
_user_specified_nameVariable_118:,(
&
_user_specified_nameVariable_117:,(
&
_user_specified_nameVariable_116:,	(
&
_user_specified_nameVariable_115:,
(
&
_user_specified_nameVariable_114:,(
&
_user_specified_nameVariable_113:,(
&
_user_specified_nameVariable_112:,(
&
_user_specified_nameVariable_111:,(
&
_user_specified_nameVariable_110:,(
&
_user_specified_nameVariable_109:,(
&
_user_specified_nameVariable_108:,(
&
_user_specified_nameVariable_107:,(
&
_user_specified_nameVariable_106:,(
&
_user_specified_nameVariable_105:,(
&
_user_specified_nameVariable_104:,(
&
_user_specified_nameVariable_103:,(
&
_user_specified_nameVariable_102:,(
&
_user_specified_nameVariable_101:,(
&
_user_specified_nameVariable_100:+'
%
_user_specified_nameVariable_99:+'
%
_user_specified_nameVariable_98:+'
%
_user_specified_nameVariable_97:+'
%
_user_specified_nameVariable_96:+'
%
_user_specified_nameVariable_95:+'
%
_user_specified_nameVariable_94:+'
%
_user_specified_nameVariable_93:+ '
%
_user_specified_nameVariable_92:+!'
%
_user_specified_nameVariable_91:+"'
%
_user_specified_nameVariable_90:+#'
%
_user_specified_nameVariable_89:+$'
%
_user_specified_nameVariable_88:+%'
%
_user_specified_nameVariable_87:+&'
%
_user_specified_nameVariable_86:+''
%
_user_specified_nameVariable_85:+('
%
_user_specified_nameVariable_84:+)'
%
_user_specified_nameVariable_83:+*'
%
_user_specified_nameVariable_82:++'
%
_user_specified_nameVariable_81:+,'
%
_user_specified_nameVariable_80:+-'
%
_user_specified_nameVariable_79:+.'
%
_user_specified_nameVariable_78:+/'
%
_user_specified_nameVariable_77:+0'
%
_user_specified_nameVariable_76:+1'
%
_user_specified_nameVariable_75:+2'
%
_user_specified_nameVariable_74:+3'
%
_user_specified_nameVariable_73:+4'
%
_user_specified_nameVariable_72:+5'
%
_user_specified_nameVariable_71:+6'
%
_user_specified_nameVariable_70:+7'
%
_user_specified_nameVariable_69:+8'
%
_user_specified_nameVariable_68:+9'
%
_user_specified_nameVariable_67:+:'
%
_user_specified_nameVariable_66:+;'
%
_user_specified_nameVariable_65:+<'
%
_user_specified_nameVariable_64:+='
%
_user_specified_nameVariable_63:+>'
%
_user_specified_nameVariable_62:+?'
%
_user_specified_nameVariable_61:+@'
%
_user_specified_nameVariable_60:+A'
%
_user_specified_nameVariable_59:+B'
%
_user_specified_nameVariable_58:+C'
%
_user_specified_nameVariable_57:+D'
%
_user_specified_nameVariable_56:+E'
%
_user_specified_nameVariable_55:+F'
%
_user_specified_nameVariable_54:+G'
%
_user_specified_nameVariable_53:+H'
%
_user_specified_nameVariable_52:+I'
%
_user_specified_nameVariable_51:+J'
%
_user_specified_nameVariable_50:+K'
%
_user_specified_nameVariable_49:+L'
%
_user_specified_nameVariable_48:+M'
%
_user_specified_nameVariable_47:+N'
%
_user_specified_nameVariable_46:+O'
%
_user_specified_nameVariable_45:+P'
%
_user_specified_nameVariable_44:+Q'
%
_user_specified_nameVariable_43:+R'
%
_user_specified_nameVariable_42:+S'
%
_user_specified_nameVariable_41:+T'
%
_user_specified_nameVariable_40:+U'
%
_user_specified_nameVariable_39:+V'
%
_user_specified_nameVariable_38:+W'
%
_user_specified_nameVariable_37:+X'
%
_user_specified_nameVariable_36:+Y'
%
_user_specified_nameVariable_35:+Z'
%
_user_specified_nameVariable_34:+['
%
_user_specified_nameVariable_33:+\'
%
_user_specified_nameVariable_32:+]'
%
_user_specified_nameVariable_31:+^'
%
_user_specified_nameVariable_30:+_'
%
_user_specified_nameVariable_29:+`'
%
_user_specified_nameVariable_28:+a'
%
_user_specified_nameVariable_27:+b'
%
_user_specified_nameVariable_26:+c'
%
_user_specified_nameVariable_25:+d'
%
_user_specified_nameVariable_24:+e'
%
_user_specified_nameVariable_23:+f'
%
_user_specified_nameVariable_22:+g'
%
_user_specified_nameVariable_21:+h'
%
_user_specified_nameVariable_20:+i'
%
_user_specified_nameVariable_19:+j'
%
_user_specified_nameVariable_18:+k'
%
_user_specified_nameVariable_17:+l'
%
_user_specified_nameVariable_16:+m'
%
_user_specified_nameVariable_15:+n'
%
_user_specified_nameVariable_14:+o'
%
_user_specified_nameVariable_13:+p'
%
_user_specified_nameVariable_12:+q'
%
_user_specified_nameVariable_11:+r'
%
_user_specified_nameVariable_10:*s&
$
_user_specified_name
Variable_9:*t&
$
_user_specified_name
Variable_8:*u&
$
_user_specified_name
Variable_7:*v&
$
_user_specified_name
Variable_6:*w&
$
_user_specified_name
Variable_5:*x&
$
_user_specified_name
Variable_4:*y&
$
_user_specified_name
Variable_3:*z&
$
_user_specified_name
Variable_2:*{&
$
_user_specified_name
Variable_1:(|$
"
_user_specified_name
Variable:W}S
Q
_user_specified_name97transformer_layer_3/self_attention_layer/query/kernel_1:U~Q
O
_user_specified_name75transformer_layer_3/self_attention_layer/value/bias_1:`\
Z
_user_specified_nameB@transformer_layer_3/self_attention_layer/attention_output/bias_1:T�O
M
_user_specified_name53transformer_layer_3/feedforward_output_dense/bias_1:U�P
N
_user_specified_name64transformer_layer_5/self_attention_layer_norm/beta_1:V�Q
O
_user_specified_name75transformer_layer_1/self_attention_layer/query/bias_1:V�Q
O
_user_specified_name75transformer_layer_3/self_attention_layer/key/kernel_1:U�P
N
_user_specified_name64transformer_layer_3/self_attention_layer_norm/beta_1:T�O
M
_user_specified_name53transformer_layer_1/self_attention_layer/key/bias_1:T�O
M
_user_specified_name53transformer_layer_4/feedforward_output_dense/bias_1:S�N
L
_user_specified_name42transformer_layer_4/feedforward_layer_norm/gamma_1:R�M
K
_user_specified_name31transformer_layer_0/feedforward_layer_norm/beta_1:V�Q
O
_user_specified_name75transformer_layer_4/self_attention_layer/query/bias_1:V�Q
O
_user_specified_name75transformer_layer_4/self_attention_layer/key/kernel_1:Z�U
S
_user_specified_name;9token_and_position_embedding/token_embedding/embeddings_1:T�O
M
_user_specified_name53transformer_layer_5/feedforward_output_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_3/self_attention_layer/query/bias_1:T�O
M
_user_specified_name53transformer_layer_3/self_attention_layer/key/bias_1:V�Q
O
_user_specified_name75transformer_layer_0/self_attention_layer_norm/gamma_1:R�M
K
_user_specified_name31transformer_layer_4/feedforward_layer_norm/beta_1:=�8
6
_user_specified_nameembeddings_layer_norm/beta_1:X�S
Q
_user_specified_name97transformer_layer_2/self_attention_layer/query/kernel_1:S�N
L
_user_specified_name42transformer_layer_2/feedforward_layer_norm/gamma_1:Z�U
S
_user_specified_name;9transformer_layer_4/feedforward_intermediate_dense/bias_1:T�O
M
_user_specified_name53transformer_layer_4/self_attention_layer/key/bias_1:X�S
Q
_user_specified_name97transformer_layer_4/self_attention_layer/value/kernel_1:V�Q
O
_user_specified_name75transformer_layer_1/feedforward_output_dense/kernel_1:c�^
\
_user_specified_nameDBtransformer_layer_0/self_attention_layer/attention_output/kernel_1:S�N
L
_user_specified_name42transformer_layer_1/feedforward_layer_norm/gamma_1:Z�U
S
_user_specified_name;9transformer_layer_1/feedforward_intermediate_dense/bias_1:4�/
-
_user_specified_namepooled_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_2/self_attention_layer/query/bias_1:R�M
K
_user_specified_name31transformer_layer_2/feedforward_layer_norm/beta_1:\�W
U
_user_specified_name=;transformer_layer_4/feedforward_intermediate_dense/kernel_1:c�^
\
_user_specified_nameDBtransformer_layer_4/self_attention_layer/attention_output/kernel_1:]�X
V
_user_specified_name><token_and_position_embedding/position_embedding/embeddings_1:V�Q
O
_user_specified_name75transformer_layer_2/self_attention_layer/key/kernel_1:Z�U
S
_user_specified_name;9transformer_layer_2/feedforward_intermediate_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_4/self_attention_layer/value/bias_1:S�N
L
_user_specified_name42transformer_layer_5/feedforward_layer_norm/gamma_1:0�+
)
_user_specified_namelogits/kernel_1:>�9
7
_user_specified_nameembeddings_layer_norm/gamma_1:X�S
Q
_user_specified_name97transformer_layer_2/self_attention_layer/value/kernel_1:S�N
L
_user_specified_name42transformer_layer_3/feedforward_layer_norm/gamma_1:V�Q
O
_user_specified_name75transformer_layer_4/self_attention_layer_norm/gamma_1:X�S
Q
_user_specified_name97transformer_layer_5/self_attention_layer/query/kernel_1:X�S
Q
_user_specified_name97transformer_layer_0/self_attention_layer/query/kernel_1:a�\
Z
_user_specified_nameB@transformer_layer_0/self_attention_layer/attention_output/bias_1:R�M
K
_user_specified_name31transformer_layer_1/feedforward_layer_norm/beta_1:V�Q
O
_user_specified_name75transformer_layer_5/self_attention_layer/key/kernel_1:V�Q
O
_user_specified_name75transformer_layer_0/self_attention_layer/key/kernel_1:U�P
N
_user_specified_name64transformer_layer_0/self_attention_layer_norm/beta_1:X�S
Q
_user_specified_name97transformer_layer_5/self_attention_layer/value/kernel_1:X�S
Q
_user_specified_name97transformer_layer_1/self_attention_layer/value/kernel_1:T�O
M
_user_specified_name53transformer_layer_2/self_attention_layer/key/bias_1:\�W
U
_user_specified_name=;transformer_layer_2/feedforward_intermediate_dense/kernel_1:a�\
Z
_user_specified_nameB@transformer_layer_4/self_attention_layer/attention_output/bias_1:R�M
K
_user_specified_name31transformer_layer_5/feedforward_layer_norm/beta_1:6�1
/
_user_specified_namepooled_dense/kernel_1:.�)
'
_user_specified_namelogits/bias_1:T�O
M
_user_specified_name53transformer_layer_0/feedforward_output_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_2/self_attention_layer/value/bias_1:c�^
\
_user_specified_nameDBtransformer_layer_2/self_attention_layer/attention_output/kernel_1:R�M
K
_user_specified_name31transformer_layer_3/feedforward_layer_norm/beta_1:U�P
N
_user_specified_name64transformer_layer_4/self_attention_layer_norm/beta_1:V�Q
O
_user_specified_name75transformer_layer_5/self_attention_layer/query/bias_1:\�W
U
_user_specified_name=;transformer_layer_5/feedforward_intermediate_dense/kernel_1:V�Q
O
_user_specified_name75transformer_layer_0/self_attention_layer/query/bias_1:V�Q
O
_user_specified_name75transformer_layer_2/self_attention_layer_norm/gamma_1:\�W
U
_user_specified_name=;transformer_layer_3/feedforward_intermediate_dense/kernel_1:T�O
M
_user_specified_name53transformer_layer_0/self_attention_layer/key/bias_1:\�W
U
_user_specified_name=;transformer_layer_1/feedforward_intermediate_dense/kernel_1:V�Q
O
_user_specified_name75transformer_layer_5/self_attention_layer/value/bias_1:c�^
\
_user_specified_nameDBtransformer_layer_5/self_attention_layer/attention_output/kernel_1:X�S
Q
_user_specified_name97transformer_layer_0/self_attention_layer/value/kernel_1:V�Q
O
_user_specified_name75transformer_layer_1/self_attention_layer/value/bias_1:c�^
\
_user_specified_nameDBtransformer_layer_1/self_attention_layer/attention_output/kernel_1:\�W
U
_user_specified_name=;transformer_layer_0/feedforward_intermediate_dense/kernel_1:V�Q
O
_user_specified_name75transformer_layer_1/self_attention_layer_norm/gamma_1:a�\
Z
_user_specified_nameB@transformer_layer_2/self_attention_layer/attention_output/bias_1:V�Q
O
_user_specified_name75transformer_layer_2/feedforward_output_dense/kernel_1:Z�U
S
_user_specified_name;9transformer_layer_5/feedforward_intermediate_dense/bias_1:U�P
N
_user_specified_name64transformer_layer_2/self_attention_layer_norm/beta_1:Z�U
S
_user_specified_name;9transformer_layer_3/feedforward_intermediate_dense/bias_1:T�O
M
_user_specified_name53transformer_layer_5/self_attention_layer/key/bias_1:T�O
M
_user_specified_name53transformer_layer_1/feedforward_output_dense/bias_1:X�S
Q
_user_specified_name97transformer_layer_3/self_attention_layer/value/kernel_1:c�^
\
_user_specified_nameDBtransformer_layer_3/self_attention_layer/attention_output/kernel_1:V�Q
O
_user_specified_name75transformer_layer_3/feedforward_output_dense/kernel_1:a�\
Z
_user_specified_nameB@transformer_layer_5/self_attention_layer/attention_output/bias_1:V�Q
O
_user_specified_name75transformer_layer_5/self_attention_layer_norm/gamma_1:V�Q
O
_user_specified_name75transformer_layer_0/self_attention_layer/value/bias_1:X�S
Q
_user_specified_name97transformer_layer_1/self_attention_layer/query/kernel_1:a�\
Z
_user_specified_nameB@transformer_layer_1/self_attention_layer/attention_output/bias_1:V�Q
O
_user_specified_name75transformer_layer_3/self_attention_layer_norm/gamma_1:Z�U
S
_user_specified_name;9transformer_layer_0/feedforward_intermediate_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_1/self_attention_layer/key/kernel_1:U�P
N
_user_specified_name64transformer_layer_1/self_attention_layer_norm/beta_1:V�Q
O
_user_specified_name75transformer_layer_4/feedforward_output_dense/kernel_1:X�S
Q
_user_specified_name97transformer_layer_4/self_attention_layer/query/kernel_1:V�Q
O
_user_specified_name75transformer_layer_0/feedforward_output_dense/kernel_1:S�N
L
_user_specified_name42transformer_layer_0/feedforward_layer_norm/gamma_1:T�O
M
_user_specified_name53transformer_layer_2/feedforward_output_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_5/feedforward_output_dense/kernel_1
�K
�
+__inference_signature_wrapper___call___8311
padding_mask
	token_ids
unknown:���
	unknown_0:
��
	unknown_1:	�
	unknown_2:	� 
	unknown_3:�@
	unknown_4:@ 
	unknown_5:�@
	unknown_6:@ 
	unknown_7:�@
	unknown_8:@ 
	unknown_9:@�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:	�!

unknown_19:�@

unknown_20:@!

unknown_21:�@

unknown_22:@!

unknown_23:�@

unknown_24:@!

unknown_25:@�

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:
��

unknown_30:	�

unknown_31:
��

unknown_32:	�

unknown_33:	�

unknown_34:	�!

unknown_35:�@

unknown_36:@!

unknown_37:�@

unknown_38:@!

unknown_39:�@

unknown_40:@!

unknown_41:@�

unknown_42:	�

unknown_43:	�

unknown_44:	�

unknown_45:
��

unknown_46:	�

unknown_47:
��

unknown_48:	�

unknown_49:	�

unknown_50:	�!

unknown_51:�@

unknown_52:@!

unknown_53:�@

unknown_54:@!

unknown_55:�@

unknown_56:@!

unknown_57:@�

unknown_58:	�

unknown_59:	�

unknown_60:	�

unknown_61:
��

unknown_62:	�

unknown_63:
��

unknown_64:	�

unknown_65:	�

unknown_66:	�!

unknown_67:�@

unknown_68:@!

unknown_69:�@

unknown_70:@!

unknown_71:�@

unknown_72:@!

unknown_73:@�

unknown_74:	�

unknown_75:	�

unknown_76:	�

unknown_77:
��

unknown_78:	�

unknown_79:
��

unknown_80:	�

unknown_81:	�

unknown_82:	�!

unknown_83:�@

unknown_84:@!

unknown_85:�@

unknown_86:@!

unknown_87:�@

unknown_88:@!

unknown_89:@�

unknown_90:	�

unknown_91:	�

unknown_92:	�

unknown_93:
��

unknown_94:	�

unknown_95:
��

unknown_96:	�

unknown_97:	�

unknown_98:	�

unknown_99:
��
unknown_100:	�
unknown_101:	�
unknown_102:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallpadding_mask	token_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92
unknown_93
unknown_94
unknown_95
unknown_96
unknown_97
unknown_98
unknown_99unknown_100unknown_101unknown_102*u
Tinn
l2j*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*�
_read_only_resource_inputsl
jh	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghi*-
config_proto

CPU

GPU 2J 8� *"
fR
__inference___call___8096o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:������������������:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:������������������
&
_user_specified_namepadding_mask:[W
0
_output_shapes
:������������������
#
_user_specified_name	token_ids:$ 

_user_specified_name8101:$ 

_user_specified_name8103:$ 

_user_specified_name8105:$ 

_user_specified_name8107:$ 

_user_specified_name8109:$ 

_user_specified_name8111:$ 

_user_specified_name8113:$	 

_user_specified_name8115:$
 

_user_specified_name8117:$ 

_user_specified_name8119:$ 

_user_specified_name8121:$ 

_user_specified_name8123:$ 

_user_specified_name8125:$ 

_user_specified_name8127:$ 

_user_specified_name8129:$ 

_user_specified_name8131:$ 

_user_specified_name8133:$ 

_user_specified_name8135:$ 

_user_specified_name8137:$ 

_user_specified_name8139:$ 

_user_specified_name8141:$ 

_user_specified_name8143:$ 

_user_specified_name8145:$ 

_user_specified_name8147:$ 

_user_specified_name8149:$ 

_user_specified_name8151:$ 

_user_specified_name8153:$ 

_user_specified_name8155:$ 

_user_specified_name8157:$ 

_user_specified_name8159:$  

_user_specified_name8161:$! 

_user_specified_name8163:$" 

_user_specified_name8165:$# 

_user_specified_name8167:$$ 

_user_specified_name8169:$% 

_user_specified_name8171:$& 

_user_specified_name8173:$' 

_user_specified_name8175:$( 

_user_specified_name8177:$) 

_user_specified_name8179:$* 

_user_specified_name8181:$+ 

_user_specified_name8183:$, 

_user_specified_name8185:$- 

_user_specified_name8187:$. 

_user_specified_name8189:$/ 

_user_specified_name8191:$0 

_user_specified_name8193:$1 

_user_specified_name8195:$2 

_user_specified_name8197:$3 

_user_specified_name8199:$4 

_user_specified_name8201:$5 

_user_specified_name8203:$6 

_user_specified_name8205:$7 

_user_specified_name8207:$8 

_user_specified_name8209:$9 

_user_specified_name8211:$: 

_user_specified_name8213:$; 

_user_specified_name8215:$< 

_user_specified_name8217:$= 

_user_specified_name8219:$> 

_user_specified_name8221:$? 

_user_specified_name8223:$@ 

_user_specified_name8225:$A 

_user_specified_name8227:$B 

_user_specified_name8229:$C 

_user_specified_name8231:$D 

_user_specified_name8233:$E 

_user_specified_name8235:$F 

_user_specified_name8237:$G 

_user_specified_name8239:$H 

_user_specified_name8241:$I 

_user_specified_name8243:$J 

_user_specified_name8245:$K 

_user_specified_name8247:$L 

_user_specified_name8249:$M 

_user_specified_name8251:$N 

_user_specified_name8253:$O 

_user_specified_name8255:$P 

_user_specified_name8257:$Q 

_user_specified_name8259:$R 

_user_specified_name8261:$S 

_user_specified_name8263:$T 

_user_specified_name8265:$U 

_user_specified_name8267:$V 

_user_specified_name8269:$W 

_user_specified_name8271:$X 

_user_specified_name8273:$Y 

_user_specified_name8275:$Z 

_user_specified_name8277:$[ 

_user_specified_name8279:$\ 

_user_specified_name8281:$] 

_user_specified_name8283:$^ 

_user_specified_name8285:$_ 

_user_specified_name8287:$` 

_user_specified_name8289:$a 

_user_specified_name8291:$b 

_user_specified_name8293:$c 

_user_specified_name8295:$d 

_user_specified_name8297:$e 

_user_specified_name8299:$f 

_user_specified_name8301:$g 

_user_specified_name8303:$h 

_user_specified_name8305:$i 

_user_specified_name8307
��
��
__inference___call___8096
padding_mask
	token_ids�
xdistil_bert_classifier_1_distil_bert_backbone_1_token_and_position_embedding_1_token_embedding_1_readvariableop_resource:����
�distil_bert_classifier_1_distil_bert_backbone_1_token_and_position_embedding_1_position_embedding_1_slice_readvariableop_resource:
��v
gdistil_bert_classifier_1_distil_bert_backbone_1_embeddings_layer_norm_1_reshape_readvariableop_resource:	�x
idistil_bert_classifier_1_distil_bert_backbone_1_embeddings_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_query_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_query_1_add_readvariableop_resource:@�
distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_key_1_cast_readvariableop_resource:�@�
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_key_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_value_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_value_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource:@��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_intermediate_dense_1_cast_readvariableop_resource:
���
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_intermediate_dense_1_add_readvariableop_resource:	��
}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_output_dense_1_cast_readvariableop_resource:
���
|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_output_dense_1_add_readvariableop_resource:	��
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_query_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_query_1_add_readvariableop_resource:@�
distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_key_1_cast_readvariableop_resource:�@�
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_key_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_value_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_value_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource:@��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_intermediate_dense_1_cast_readvariableop_resource:
���
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_intermediate_dense_1_add_readvariableop_resource:	��
}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_output_dense_1_cast_readvariableop_resource:
���
|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_output_dense_1_add_readvariableop_resource:	��
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_query_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_query_1_add_readvariableop_resource:@�
distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_key_1_cast_readvariableop_resource:�@�
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_key_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_value_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_value_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource:@��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_intermediate_dense_1_cast_readvariableop_resource:
���
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_intermediate_dense_1_add_readvariableop_resource:	��
}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_output_dense_1_cast_readvariableop_resource:
���
|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_output_dense_1_add_readvariableop_resource:	��
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_query_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_query_1_add_readvariableop_resource:@�
distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_key_1_cast_readvariableop_resource:�@�
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_key_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_value_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_value_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource:@��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_intermediate_dense_1_cast_readvariableop_resource:
���
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_intermediate_dense_1_add_readvariableop_resource:	��
}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_output_dense_1_cast_readvariableop_resource:
���
|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_output_dense_1_add_readvariableop_resource:	��
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_query_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_query_1_add_readvariableop_resource:@�
distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_key_1_cast_readvariableop_resource:�@�
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_key_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_value_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_value_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource:@��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_intermediate_dense_1_cast_readvariableop_resource:
���
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_intermediate_dense_1_add_readvariableop_resource:	��
}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_output_dense_1_cast_readvariableop_resource:
���
|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_output_dense_1_add_readvariableop_resource:	��
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_query_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_query_1_add_readvariableop_resource:@�
distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_key_1_cast_readvariableop_resource:�@�
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_key_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_value_1_cast_readvariableop_resource:�@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_value_1_add_readvariableop_resource:@�
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource:@��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_intermediate_dense_1_cast_readvariableop_resource:
���
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_intermediate_dense_1_add_readvariableop_resource:	��
}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_output_dense_1_cast_readvariableop_resource:
���
|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_output_dense_1_add_readvariableop_resource:	��
~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_layer_norm_1_reshape_readvariableop_resource:	��
�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource:	�X
Ddistil_bert_classifier_1_pooled_dense_1_cast_readvariableop_resource:
��R
Cdistil_bert_classifier_1_pooled_dense_1_add_readvariableop_resource:	�Q
>distil_bert_classifier_1_logits_1_cast_readvariableop_resource:	�K
=distil_bert_classifier_1_logits_1_add_readvariableop_resource:
identity��^distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape/ReadVariableOp�`distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape_1/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/ReadVariableOp�odistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/ReadVariableOp�ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Add/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp�sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Add/ReadVariableOp�tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp�ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Add/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp�sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Add/ReadVariableOp�tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp�ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Add/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp�sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Add/ReadVariableOp�tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp�ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Add/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp�sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Add/ReadVariableOp�tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp�ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Add/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp�sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Add/ReadVariableOp�tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp�ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Add/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp�sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Add/ReadVariableOp�tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp��distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/Cast/ReadVariableOp�udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/Cast/ReadVariableOp�wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/add/ReadVariableOp�xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape/ReadVariableOp�zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp�4distil_bert_classifier_1/logits_1/Add/ReadVariableOp�5distil_bert_classifier_1/logits_1/Cast/ReadVariableOp�:distil_bert_classifier_1/pooled_dense_1/Add/ReadVariableOp�;distil_bert_classifier_1/pooled_dense_1/Cast/ReadVariableOp�
odistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/ReadVariableOpReadVariableOpxdistil_bert_classifier_1_distil_bert_backbone_1_token_and_position_embedding_1_token_embedding_1_readvariableop_resource*!
_output_shapes
:���*
dtype0�
fdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/ShapeConst*
_output_shapes
:*
dtype0*
valueB":w     �
tdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
vdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
vdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
ndistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/strided_sliceStridedSliceodistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/Shape:output:0}distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/strided_slice/stack:output:0distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/strided_slice/stack_1:output:0distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
fdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
ddistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/subSubwdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/strided_slice:output:0odistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/sub/y:output:0*
T0*
_output_shapes
: �
vdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/clip_by_value/MinimumMinimum	token_idshdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/sub:z:0*
T0*0
_output_shapes
:�������������������
pdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : �
ndistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/clip_by_valueMaximumzdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/clip_by_value/Minimum:z:0ydistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/clip_by_value/y:output:0*
T0*0
_output_shapes
:�������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
idistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/GatherV2GatherV2wdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/ReadVariableOp:value:0rdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/clip_by_value:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*5
_output_shapes#
!:��������������������
idistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/ShapeShaperdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/GatherV2:output:0*
T0*
_output_shapes
::���
wdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
ydistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
ydistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
qdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_sliceStridedSlicerdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Shape:output:0�distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice/stack:output:0�distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice/stack_1:output:0�distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
ydistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:�
{distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
{distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
sdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice_1StridedSlicerdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Shape:output:0�distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice_1/stack:output:0�distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice_1/stack_1:output:0�distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
xdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_token_and_position_embedding_1_position_embedding_1_slice_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
odistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB"        �
pdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/size/1Const*
_output_shapes
: *
dtype0*
value
B :��
ndistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/sizePack|distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice_1:output:0ydistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/size/1:output:0*
N*
T0*
_output_shapes
:�
idistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/SliceSlice�distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/ReadVariableOp:value:0xdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/begin:output:0wdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/size:output:0*
Index0*
T0*(
_output_shapes
:�����������
wdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value
B :��
udistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/BroadcastTo/shapePackzdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice:output:0|distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/strided_slice_1:output:0�distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:�
odistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/BroadcastToBroadcastTordistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice:output:0~distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/BroadcastTo/shape:output:0*
T0*5
_output_shapes#
!:��������������������
Rdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/addAddV2rdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/GatherV2:output:0xdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/BroadcastTo:output:0*
T0*5
_output_shapes#
!:��������������������
fdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
Tdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/meanMeanVdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/add:z:0odistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
\distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/StopGradientStopGradient]distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
adistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceVdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/add:z:0edistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
jdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
Xdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/varianceMeanedistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/SquaredDifference:z:0sdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
^distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape/ReadVariableOpReadVariableOpgdistil_bert_classifier_1_distil_bert_backbone_1_embeddings_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Udistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
Odistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/ReshapeReshapefdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape/ReadVariableOp:value:0^distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
`distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOpidistil_bert_classifier_1_distil_bert_backbone_1_embeddings_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Wdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
Qdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape_1Reshapehdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape_1/ReadVariableOp:value:0`distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
Mdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
Kdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/addAddV2adistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/variance:output:0Vdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
Mdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/RsqrtRsqrtOdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
Kdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/mulMulQdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Rsqrt:y:0Xdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
Kdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/NegNeg]distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
Mdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/mul_1MulOdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Neg:y:0Odistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Mdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/add_1AddV2Qdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/mul_1:z:0Zdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
Mdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/mul_2MulVdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/add:z:0Odistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
Mdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/add_2AddV2Qdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/mul_2:z:0Qdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/ExpandDims
ExpandDimspadding_mask]distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_query_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/einsum/EinsumEinsumQdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_query_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/Cast/ReadVariableOpReadVariableOpdistil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_key_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/einsum/EinsumEinsumQdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/add_2:z:0~distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/add/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_key_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/addAddV2ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/einsum/Einsum:output:0}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_value_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/einsum/EinsumEinsumQdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_value_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
`distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/MulMulldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/add:z:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/Cast/x:output:0*
T0*8
_output_shapes&
$:"������������������@�
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose	Transposejdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/add:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_1	Transposeddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/Mul:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"���������@����������
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/MatMulBatchMatMulV2jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose:y:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_1:y:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_2	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/MatMul:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_2/perm:output:0*
T0*A
_output_shapes/
-:+����������������������������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/ExpandDims
ExpandDimsYdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/ExpandDims:output:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"�������������������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/CastCastpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/ExpandDims:output:0*

DstT0*

SrcT0*8
_output_shapes&
$:"�������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/subSubudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/sub/x:output:0odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/Cast:y:0*
T0*8
_output_shapes&
$:"�������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/mulMulndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/sub:z:0udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/mul/y:output:0*
T0*8
_output_shapes&
$:"�������������������
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/addAddV2ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_2:y:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/SoftmaxSoftmaxndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_3	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/add:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/MatMul_1BatchMatMulV2xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/softmax_1/Softmax:softmax:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_3:y:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_4	Transposendistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/MatMul_1:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_4/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/einsum/EinsumEinsumldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/transpose_4:y:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/addAddV2�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/einsum/Einsum:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/addAddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/add:z:0Qdistil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/meanMeanMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/add:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/StopGradientStopGradientwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/add:z:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/varianceMeandistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/ReshapeReshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape/ReadVariableOp:value:0xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape_1Reshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp:value:0zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/variance:output:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/RsqrtRsqrtidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/mulMulkdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Rsqrt:y:0rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/NegNegwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/mul_1Mulidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Neg:y:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/add_1AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/mul_1:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/mul_2MulMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/add:z:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/add_2AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/mul_2:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_intermediate_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/MatMulBatchMatMulV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_intermediate_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/AddAddV2vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/MatMul:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/mulMuludistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Const:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Add:z:0*
T0*5
_output_shapes#
!:��������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/truedivRealDivndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Add:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Const_2:output:0*
T0*5
_output_shapes#
!:��������������������
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/ErfErfrdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/truediv:z:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/add_1AddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Const_1:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Erf:y:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/mul_1Mulndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/mul:z:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Cast/ReadVariableOpReadVariableOp}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_output_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/MatMulBatchMatMulV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/mul_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Add/ReadVariableOpReadVariableOp|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_output_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/AddAddV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/MatMul:output:0{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Add:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/meanMeanOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/add_1:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/StopGradientStopGradienttdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/add_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/varianceMean|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/ReshapeReshape}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape/ReadVariableOp:value:0udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_0_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape_1Reshapedistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp:value:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/addAddV2xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/variance:output:0mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/RsqrtRsqrtfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/mulMulhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Rsqrt:y:0odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/NegNegtdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/mul_1Mulfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Neg:y:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/mul_1:z:0qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/mul_2MulOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/add_1:z:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/add_2AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/mul_2:z:0hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/ExpandDims
ExpandDimspadding_mask]distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_query_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_query_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/Cast/ReadVariableOpReadVariableOpdistil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_key_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/add_2:z:0~distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/add/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_key_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/addAddV2ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/einsum/Einsum:output:0}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_value_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_value_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
`distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/MulMulldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/add:z:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/Cast/x:output:0*
T0*8
_output_shapes&
$:"������������������@�
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose	Transposejdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/add:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_1	Transposeddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/Mul:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"���������@����������
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/MatMulBatchMatMulV2jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose:y:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_1:y:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_2	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/MatMul:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_2/perm:output:0*
T0*A
_output_shapes/
-:+����������������������������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/ExpandDims
ExpandDimsYdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/ExpandDims:output:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"�������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/CastCastpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/ExpandDims:output:0*

DstT0*

SrcT0*8
_output_shapes&
$:"�������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/subSubwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/sub/x:output:0qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/Cast:y:0*
T0*8
_output_shapes&
$:"�������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/mulMulpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/sub:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/mul/y:output:0*
T0*8
_output_shapes&
$:"�������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/addAddV2ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_2:y:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/SoftmaxSoftmaxpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_3	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/add:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/MatMul_1BatchMatMulV2zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/softmax_1_1/Softmax:softmax:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_3:y:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_4	Transposendistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/MatMul_1:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_4/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/einsum/EinsumEinsumldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/transpose_4:y:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/addAddV2�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/einsum/Einsum:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/addAddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/add:z:0hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/meanMeanMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/add:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/StopGradientStopGradientwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/add:z:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/varianceMeandistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/ReshapeReshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape/ReadVariableOp:value:0xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape_1Reshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp:value:0zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/variance:output:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/RsqrtRsqrtidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/mulMulkdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Rsqrt:y:0rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/NegNegwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/mul_1Mulidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Neg:y:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/add_1AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/mul_1:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/mul_2MulMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/add:z:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/add_2AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/mul_2:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_intermediate_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/MatMulBatchMatMulV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_intermediate_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/AddAddV2vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/MatMul:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/mulMuludistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Const:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Add:z:0*
T0*5
_output_shapes#
!:��������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/truedivRealDivndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Add:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Const_2:output:0*
T0*5
_output_shapes#
!:��������������������
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/ErfErfrdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/truediv:z:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/add_1AddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Const_1:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Erf:y:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/mul_1Mulndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/mul:z:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Cast/ReadVariableOpReadVariableOp}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_output_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/MatMulBatchMatMulV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/mul_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Add/ReadVariableOpReadVariableOp|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_output_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/AddAddV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/MatMul:output:0{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Add:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/meanMeanOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/add_1:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/StopGradientStopGradienttdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/add_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/varianceMean|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/ReshapeReshape}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape/ReadVariableOp:value:0udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_1_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape_1Reshapedistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp:value:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/addAddV2xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/variance:output:0mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/RsqrtRsqrtfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/mulMulhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Rsqrt:y:0odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/NegNegtdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/mul_1Mulfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Neg:y:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/mul_1:z:0qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/mul_2MulOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/add_1:z:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/add_2AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/mul_2:z:0hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/ExpandDims
ExpandDimspadding_mask]distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_query_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_query_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/Cast/ReadVariableOpReadVariableOpdistil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_key_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/add_2:z:0~distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/add/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_key_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/addAddV2ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/einsum/Einsum:output:0}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_value_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_value_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
`distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/MulMulldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/add:z:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/Cast/x:output:0*
T0*8
_output_shapes&
$:"������������������@�
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose	Transposejdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/add:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_1	Transposeddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/Mul:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"���������@����������
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/MatMulBatchMatMulV2jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose:y:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_1:y:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_2	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/MatMul:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_2/perm:output:0*
T0*A
_output_shapes/
-:+����������������������������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/ExpandDims
ExpandDimsYdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/ExpandDims:output:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"�������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/CastCastpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/ExpandDims:output:0*

DstT0*

SrcT0*8
_output_shapes&
$:"�������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/subSubwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/sub/x:output:0qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/Cast:y:0*
T0*8
_output_shapes&
$:"�������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/mulMulpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/sub:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/mul/y:output:0*
T0*8
_output_shapes&
$:"�������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/addAddV2ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_2:y:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/SoftmaxSoftmaxpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_3	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/add:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/MatMul_1BatchMatMulV2zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/softmax_2_1/Softmax:softmax:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_3:y:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_4	Transposendistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/MatMul_1:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_4/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/einsum/EinsumEinsumldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/transpose_4:y:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/addAddV2�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/einsum/Einsum:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/addAddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/add:z:0hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/meanMeanMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/add:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/StopGradientStopGradientwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/add:z:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/varianceMeandistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/ReshapeReshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape/ReadVariableOp:value:0xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape_1Reshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp:value:0zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/variance:output:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/RsqrtRsqrtidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/mulMulkdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Rsqrt:y:0rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/NegNegwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/mul_1Mulidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Neg:y:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/add_1AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/mul_1:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/mul_2MulMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/add:z:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/add_2AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/mul_2:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_intermediate_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/MatMulBatchMatMulV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_intermediate_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/AddAddV2vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/MatMul:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/mulMuludistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Const:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Add:z:0*
T0*5
_output_shapes#
!:��������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/truedivRealDivndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Add:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Const_2:output:0*
T0*5
_output_shapes#
!:��������������������
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/ErfErfrdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/truediv:z:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/add_1AddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Const_1:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Erf:y:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/mul_1Mulndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/mul:z:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Cast/ReadVariableOpReadVariableOp}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_output_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/MatMulBatchMatMulV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/mul_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Add/ReadVariableOpReadVariableOp|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_output_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/AddAddV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/MatMul:output:0{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Add:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/meanMeanOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/add_1:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/StopGradientStopGradienttdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/add_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/varianceMean|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/ReshapeReshape}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape/ReadVariableOp:value:0udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_2_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape_1Reshapedistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp:value:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/addAddV2xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/variance:output:0mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/RsqrtRsqrtfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/mulMulhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Rsqrt:y:0odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/NegNegtdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/mul_1Mulfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Neg:y:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/mul_1:z:0qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/mul_2MulOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/add_1:z:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/add_2AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/mul_2:z:0hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/ExpandDims
ExpandDimspadding_mask]distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_query_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_query_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/Cast/ReadVariableOpReadVariableOpdistil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_key_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/add_2:z:0~distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/add/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_key_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/addAddV2ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/einsum/Einsum:output:0}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_value_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_value_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
`distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/MulMulldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/add:z:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/Cast/x:output:0*
T0*8
_output_shapes&
$:"������������������@�
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose	Transposejdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/add:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_1	Transposeddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/Mul:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"���������@����������
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/MatMulBatchMatMulV2jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose:y:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_1:y:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_2	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/MatMul:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_2/perm:output:0*
T0*A
_output_shapes/
-:+����������������������������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/ExpandDims
ExpandDimsYdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/ExpandDims:output:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"�������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/CastCastpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/ExpandDims:output:0*

DstT0*

SrcT0*8
_output_shapes&
$:"�������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/subSubwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/sub/x:output:0qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/Cast:y:0*
T0*8
_output_shapes&
$:"�������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/mulMulpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/sub:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/mul/y:output:0*
T0*8
_output_shapes&
$:"�������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/addAddV2ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_2:y:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/SoftmaxSoftmaxpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_3	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/add:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/MatMul_1BatchMatMulV2zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/softmax_3_1/Softmax:softmax:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_3:y:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_4	Transposendistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/MatMul_1:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_4/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/einsum/EinsumEinsumldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/transpose_4:y:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/addAddV2�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/einsum/Einsum:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/addAddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/add:z:0hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/meanMeanMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/add:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/StopGradientStopGradientwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/add:z:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/varianceMeandistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/ReshapeReshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape/ReadVariableOp:value:0xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape_1Reshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp:value:0zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/variance:output:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/RsqrtRsqrtidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/mulMulkdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Rsqrt:y:0rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/NegNegwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/mul_1Mulidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Neg:y:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/add_1AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/mul_1:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/mul_2MulMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/add:z:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/add_2AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/mul_2:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_intermediate_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/MatMulBatchMatMulV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_intermediate_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/AddAddV2vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/MatMul:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/mulMuludistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Const:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Add:z:0*
T0*5
_output_shapes#
!:��������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/truedivRealDivndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Add:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Const_2:output:0*
T0*5
_output_shapes#
!:��������������������
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/ErfErfrdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/truediv:z:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/add_1AddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Const_1:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Erf:y:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/mul_1Mulndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/mul:z:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Cast/ReadVariableOpReadVariableOp}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_output_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/MatMulBatchMatMulV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/mul_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Add/ReadVariableOpReadVariableOp|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_output_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/AddAddV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/MatMul:output:0{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Add:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/meanMeanOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/add_1:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/StopGradientStopGradienttdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/add_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/varianceMean|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/ReshapeReshape}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape/ReadVariableOp:value:0udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_3_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape_1Reshapedistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp:value:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/addAddV2xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/variance:output:0mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/RsqrtRsqrtfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/mulMulhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Rsqrt:y:0odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/NegNegtdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/mul_1Mulfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Neg:y:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/mul_1:z:0qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/mul_2MulOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/add_1:z:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/add_2AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/mul_2:z:0hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/ExpandDims
ExpandDimspadding_mask]distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_query_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_query_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/Cast/ReadVariableOpReadVariableOpdistil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_key_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/add_2:z:0~distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/add/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_key_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/addAddV2ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/einsum/Einsum:output:0}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_value_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_value_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
`distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/MulMulldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/add:z:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/Cast/x:output:0*
T0*8
_output_shapes&
$:"������������������@�
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose	Transposejdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/add:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_1	Transposeddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/Mul:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"���������@����������
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/MatMulBatchMatMulV2jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose:y:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_1:y:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_2	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/MatMul:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_2/perm:output:0*
T0*A
_output_shapes/
-:+����������������������������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/ExpandDims
ExpandDimsYdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/ExpandDims:output:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"�������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/CastCastpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/ExpandDims:output:0*

DstT0*

SrcT0*8
_output_shapes&
$:"�������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/subSubwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/sub/x:output:0qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/Cast:y:0*
T0*8
_output_shapes&
$:"�������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/mulMulpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/sub:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/mul/y:output:0*
T0*8
_output_shapes&
$:"�������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/addAddV2ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_2:y:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/SoftmaxSoftmaxpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_3	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/add:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/MatMul_1BatchMatMulV2zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/softmax_4_1/Softmax:softmax:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_3:y:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_4	Transposendistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/MatMul_1:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_4/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/einsum/EinsumEinsumldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/transpose_4:y:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/addAddV2�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/einsum/Einsum:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/addAddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/add:z:0hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/meanMeanMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/add:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/StopGradientStopGradientwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/add:z:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/varianceMeandistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/ReshapeReshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape/ReadVariableOp:value:0xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape_1Reshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp:value:0zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/variance:output:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/RsqrtRsqrtidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/mulMulkdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Rsqrt:y:0rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/NegNegwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/mul_1Mulidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Neg:y:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/add_1AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/mul_1:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/mul_2MulMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/add:z:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/add_2AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/mul_2:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_intermediate_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/MatMulBatchMatMulV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_intermediate_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/AddAddV2vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/MatMul:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/mulMuludistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Const:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Add:z:0*
T0*5
_output_shapes#
!:��������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/truedivRealDivndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Add:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Const_2:output:0*
T0*5
_output_shapes#
!:��������������������
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/ErfErfrdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/truediv:z:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/add_1AddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Const_1:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Erf:y:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/mul_1Mulndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/mul:z:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Cast/ReadVariableOpReadVariableOp}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_output_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/MatMulBatchMatMulV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/mul_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Add/ReadVariableOpReadVariableOp|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_output_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/AddAddV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/MatMul:output:0{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Add:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/meanMeanOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/add_1:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/StopGradientStopGradienttdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/add_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/varianceMean|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/ReshapeReshape}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape/ReadVariableOp:value:0udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_4_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape_1Reshapedistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp:value:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/addAddV2xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/variance:output:0mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/RsqrtRsqrtfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/mulMulhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Rsqrt:y:0odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/NegNegtdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/mul_1Mulfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Neg:y:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/mul_1:z:0qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/mul_2MulOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/add_1:z:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/add_2AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/mul_2:z:0hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
Tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/ExpandDims
ExpandDimspadding_mask]distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/ExpandDims/dim:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_query_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_query_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/Cast/ReadVariableOpReadVariableOpdistil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_key_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/add_2:z:0~distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/add/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_key_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/addAddV2ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/einsum/Einsum:output:0}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_value_1_cast_readvariableop_resource*#
_output_shapes
:�@*
dtype0�
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/einsum/EinsumEinsumhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/Cast/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_value_1_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/einsum/Einsum:output:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
`distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/MulMulldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/add:z:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/Cast/x:output:0*
T0*8
_output_shapes&
$:"������������������@�
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose	Transposejdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/add:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_1	Transposeddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/Mul:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"���������@����������
cdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/MatMulBatchMatMulV2jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose:y:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_1:y:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_2	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/MatMul:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_2/perm:output:0*
T0*A
_output_shapes/
-:+����������������������������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/ExpandDims
ExpandDimsYdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/ExpandDims:output:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"�������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/CastCastpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/ExpandDims:output:0*

DstT0*

SrcT0*8
_output_shapes&
$:"�������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/subSubwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/sub/x:output:0qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/Cast:y:0*
T0*8
_output_shapes&
$:"�������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn��
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/mulMulpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/sub:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/mul/y:output:0*
T0*8
_output_shapes&
$:"�������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/addAddV2ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_2:y:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/mul:z:0*
T0*A
_output_shapes/
-:+����������������������������
pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/SoftmaxSoftmaxpdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/add:z:0*
T0*A
_output_shapes/
-:+����������������������������
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_3	Transposeldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/add:z:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/MatMul_1BatchMatMulV2zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/softmax_5_1/Softmax:softmax:0ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_3:y:0*
T0*8
_output_shapes&
$:"������������������@�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_4/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_4	Transposendistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/MatMul_1:output:0vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_4/perm:output:0*
T0*8
_output_shapes&
$:"������������������@�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_attention_output_1_cast_readvariableop_resource*#
_output_shapes
:@�*
dtype0�
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/einsum/EinsumEinsumldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/transpose_4:y:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp:value:0*
N*
T0*5
_output_shapes#
!:�������������������*
equationabcd,cde->abe�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_1_attention_output_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/addAddV2�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/einsum/Einsum:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/addAddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/add:z:0hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/meanMeanMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/add:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/StopGradientStopGradientwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/add:z:0distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/varianceMeandistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/ReshapeReshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape/ReadVariableOp:value:0xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_self_attention_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape_1Reshape�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp:value:0zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/addAddV2{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/variance:output:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/RsqrtRsqrtidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/mulMulkdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Rsqrt:y:0rdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
edistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/NegNegwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/mul_1Mulidistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Neg:y:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/add_1AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/mul_1:z:0tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/mul_2MulMdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/add:z:0idistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/add_2AddV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/mul_2:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_intermediate_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/MatMulBatchMatMulV2kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/add_2:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Add/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_intermediate_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/AddAddV2vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/MatMul:output:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/mulMuludistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Const:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Add:z:0*
T0*5
_output_shapes#
!:��������������������
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *��?�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/truedivRealDivndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Add:z:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Const_2:output:0*
T0*5
_output_shapes#
!:��������������������
jdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/ErfErfrdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/truediv:z:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/add_1AddV2wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Const_1:output:0ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Erf:y:0*
T0*5
_output_shapes#
!:��������������������
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/mul_1Mulndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/mul:z:0pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Cast/ReadVariableOpReadVariableOp}distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_output_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/MatMulBatchMatMulV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/mul_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Cast/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Add/ReadVariableOpReadVariableOp|distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_output_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/AddAddV2pdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/MatMul:output:0{distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Add/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:��������������������
Kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Add:z:0kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/add_2:z:0*
T0*5
_output_shapes#
!:��������������������
}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
kdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/meanMeanOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/add_1:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/StopGradientStopGradienttdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/SquaredDifferenceSquaredDifferenceOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/add_1:z:0|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:��������������������
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/varianceMean|distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/SquaredDifference:z:0�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/variance/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������*
	keep_dims(�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape/ReadVariableOpReadVariableOp~distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_layer_norm_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ldistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/ReshapeReshape}distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape/ReadVariableOp:value:0udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape/shape:output:0*
T0*#
_output_shapes
:��
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpReadVariableOp�distil_bert_classifier_1_distil_bert_backbone_1_transformer_layer_5_1_feedforward_layer_norm_1_reshape_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ndistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape_1Reshapedistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp:value:0wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape_1/shape:output:0*
T0*#
_output_shapes
:��
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/addAddV2xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/variance:output:0mdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/add/y:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/RsqrtRsqrtfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/add:z:0*
T0*4
_output_shapes"
 :�������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/mulMulhdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Rsqrt:y:0odistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape:output:0*
T0*5
_output_shapes#
!:��������������������
bdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/NegNegtdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/moments/mean:output:0*
T0*4
_output_shapes"
 :�������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/mul_1Mulfdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Neg:y:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/add_1AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/mul_1:z:0qdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape_1:output:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/mul_2MulOdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/add_1:z:0fdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/mul:z:0*
T0*5
_output_shapes#
!:��������������������
ddistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/add_2AddV2hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/mul_2:z:0hdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/add_1:z:0*
T0*5
_output_shapes#
!:��������������������
,distil_bert_classifier_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
.distil_bert_classifier_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
.distil_bert_classifier_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
&distil_bert_classifier_1/strided_sliceStridedSlicehdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/add_2:z:05distil_bert_classifier_1/strided_slice/stack:output:07distil_bert_classifier_1/strided_slice/stack_1:output:07distil_bert_classifier_1/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask�
;distil_bert_classifier_1/pooled_dense_1/Cast/ReadVariableOpReadVariableOpDdistil_bert_classifier_1_pooled_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
.distil_bert_classifier_1/pooled_dense_1/MatMulMatMul/distil_bert_classifier_1/strided_slice:output:0Cdistil_bert_classifier_1/pooled_dense_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:distil_bert_classifier_1/pooled_dense_1/Add/ReadVariableOpReadVariableOpCdistil_bert_classifier_1_pooled_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+distil_bert_classifier_1/pooled_dense_1/AddAddV28distil_bert_classifier_1/pooled_dense_1/MatMul:product:0Bdistil_bert_classifier_1/pooled_dense_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,distil_bert_classifier_1/pooled_dense_1/ReluRelu/distil_bert_classifier_1/pooled_dense_1/Add:z:0*
T0*(
_output_shapes
:�����������
5distil_bert_classifier_1/logits_1/Cast/ReadVariableOpReadVariableOp>distil_bert_classifier_1_logits_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
(distil_bert_classifier_1/logits_1/MatMulMatMul:distil_bert_classifier_1/pooled_dense_1/Relu:activations:0=distil_bert_classifier_1/logits_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
4distil_bert_classifier_1/logits_1/Add/ReadVariableOpReadVariableOp=distil_bert_classifier_1_logits_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
%distil_bert_classifier_1/logits_1/AddAddV22distil_bert_classifier_1/logits_1/MatMul:product:0<distil_bert_classifier_1/logits_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
IdentityIdentity)distil_bert_classifier_1/logits_1/Add:z:0^NoOp*
T0*'
_output_shapes
:����������b
NoOpNoOp_^distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape/ReadVariableOpa^distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape_1/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/ReadVariableOpp^distil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/ReadVariableOpz^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Add/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpt^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Add/ReadVariableOpu^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpw^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpz^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Add/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpt^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Add/ReadVariableOpu^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpw^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpz^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Add/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpt^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Add/ReadVariableOpu^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpw^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpz^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Add/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpt^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Add/ReadVariableOpu^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpw^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpz^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Add/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpt^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Add/ReadVariableOpu^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpw^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpz^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Add/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpt^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Add/ReadVariableOpu^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/add/ReadVariableOpw^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/Cast/ReadVariableOpv^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/Cast/ReadVariableOpx^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/add/ReadVariableOpy^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape/ReadVariableOp{^distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp5^distil_bert_classifier_1/logits_1/Add/ReadVariableOp6^distil_bert_classifier_1/logits_1/Cast/ReadVariableOp;^distil_bert_classifier_1/pooled_dense_1/Add/ReadVariableOp<^distil_bert_classifier_1/pooled_dense_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:������������������:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2�
^distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape/ReadVariableOp^distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape/ReadVariableOp2�
`distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape_1/ReadVariableOp`distil_bert_classifier_1/distil_bert_backbone_1/embeddings_layer_norm_1/Reshape_1/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/position_embedding_1/Slice/ReadVariableOp2�
odistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/ReadVariableOpodistil_bert_classifier_1/distil_bert_backbone_1/token_and_position_embedding_1/token_embedding_1/ReadVariableOp2�
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Add/ReadVariableOpydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Add/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp2�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Add/ReadVariableOpsdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Add/ReadVariableOp2�
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Cast/ReadVariableOptdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/feedforward_output_dense_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp2�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/Cast/ReadVariableOpvdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/add/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/key_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/query_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_1/value_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_0_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp2�
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Add/ReadVariableOpydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Add/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp2�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Add/ReadVariableOpsdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Add/ReadVariableOp2�
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Cast/ReadVariableOptdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/feedforward_output_dense_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp2�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/Cast/ReadVariableOpvdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/add/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/key_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/query_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_1/value_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_1_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp2�
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Add/ReadVariableOpydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Add/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp2�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Add/ReadVariableOpsdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Add/ReadVariableOp2�
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Cast/ReadVariableOptdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/feedforward_output_dense_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp2�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/Cast/ReadVariableOpvdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/add/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/key_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/query_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_1/value_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_2_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp2�
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Add/ReadVariableOpydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Add/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp2�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Add/ReadVariableOpsdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Add/ReadVariableOp2�
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Cast/ReadVariableOptdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/feedforward_output_dense_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp2�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/Cast/ReadVariableOpvdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/add/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/key_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/query_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_1/value_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_3_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp2�
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Add/ReadVariableOpydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Add/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp2�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Add/ReadVariableOpsdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Add/ReadVariableOp2�
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Cast/ReadVariableOptdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/feedforward_output_dense_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp2�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/Cast/ReadVariableOpvdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/add/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/key_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/query_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_1/value_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_4_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp2�
ydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Add/ReadVariableOpydistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Add/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Cast/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_intermediate_dense_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_layer_norm_1/Reshape_1/ReadVariableOp2�
sdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Add/ReadVariableOpsdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Add/ReadVariableOp2�
tdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Cast/ReadVariableOptdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/feedforward_output_dense_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/Cast/ReadVariableOp2�
�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp�distil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/attention_output_1/add/ReadVariableOp2�
vdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/Cast/ReadVariableOpvdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/Cast/ReadVariableOp2�
udistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/add/ReadVariableOpudistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/key_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/query_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/Cast/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/Cast/ReadVariableOp2�
wdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/add/ReadVariableOpwdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_1/value_1/add/ReadVariableOp2�
xdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape/ReadVariableOpxdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape/ReadVariableOp2�
zdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOpzdistil_bert_classifier_1/distil_bert_backbone_1/transformer_layer_5_1/self_attention_layer_norm_1/Reshape_1/ReadVariableOp2l
4distil_bert_classifier_1/logits_1/Add/ReadVariableOp4distil_bert_classifier_1/logits_1/Add/ReadVariableOp2n
5distil_bert_classifier_1/logits_1/Cast/ReadVariableOp5distil_bert_classifier_1/logits_1/Cast/ReadVariableOp2x
:distil_bert_classifier_1/pooled_dense_1/Add/ReadVariableOp:distil_bert_classifier_1/pooled_dense_1/Add/ReadVariableOp2z
;distil_bert_classifier_1/pooled_dense_1/Cast/ReadVariableOp;distil_bert_classifier_1/pooled_dense_1/Cast/ReadVariableOp:^ Z
0
_output_shapes
:������������������
&
_user_specified_namepadding_mask:[W
0
_output_shapes
:������������������
#
_user_specified_name	token_ids:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource:()$
"
_user_specified_name
resource:(*$
"
_user_specified_name
resource:(+$
"
_user_specified_name
resource:(,$
"
_user_specified_name
resource:(-$
"
_user_specified_name
resource:(.$
"
_user_specified_name
resource:(/$
"
_user_specified_name
resource:(0$
"
_user_specified_name
resource:(1$
"
_user_specified_name
resource:(2$
"
_user_specified_name
resource:(3$
"
_user_specified_name
resource:(4$
"
_user_specified_name
resource:(5$
"
_user_specified_name
resource:(6$
"
_user_specified_name
resource:(7$
"
_user_specified_name
resource:(8$
"
_user_specified_name
resource:(9$
"
_user_specified_name
resource:(:$
"
_user_specified_name
resource:(;$
"
_user_specified_name
resource:(<$
"
_user_specified_name
resource:(=$
"
_user_specified_name
resource:(>$
"
_user_specified_name
resource:(?$
"
_user_specified_name
resource:(@$
"
_user_specified_name
resource:(A$
"
_user_specified_name
resource:(B$
"
_user_specified_name
resource:(C$
"
_user_specified_name
resource:(D$
"
_user_specified_name
resource:(E$
"
_user_specified_name
resource:(F$
"
_user_specified_name
resource:(G$
"
_user_specified_name
resource:(H$
"
_user_specified_name
resource:(I$
"
_user_specified_name
resource:(J$
"
_user_specified_name
resource:(K$
"
_user_specified_name
resource:(L$
"
_user_specified_name
resource:(M$
"
_user_specified_name
resource:(N$
"
_user_specified_name
resource:(O$
"
_user_specified_name
resource:(P$
"
_user_specified_name
resource:(Q$
"
_user_specified_name
resource:(R$
"
_user_specified_name
resource:(S$
"
_user_specified_name
resource:(T$
"
_user_specified_name
resource:(U$
"
_user_specified_name
resource:(V$
"
_user_specified_name
resource:(W$
"
_user_specified_name
resource:(X$
"
_user_specified_name
resource:(Y$
"
_user_specified_name
resource:(Z$
"
_user_specified_name
resource:([$
"
_user_specified_name
resource:(\$
"
_user_specified_name
resource:(]$
"
_user_specified_name
resource:(^$
"
_user_specified_name
resource:(_$
"
_user_specified_name
resource:(`$
"
_user_specified_name
resource:(a$
"
_user_specified_name
resource:(b$
"
_user_specified_name
resource:(c$
"
_user_specified_name
resource:(d$
"
_user_specified_name
resource:(e$
"
_user_specified_name
resource:(f$
"
_user_specified_name
resource:(g$
"
_user_specified_name
resource:(h$
"
_user_specified_name
resource:(i$
"
_user_specified_name
resource
�K
�
+__inference_signature_wrapper___call___8525
padding_mask
	token_ids
unknown:���
	unknown_0:
��
	unknown_1:	�
	unknown_2:	� 
	unknown_3:�@
	unknown_4:@ 
	unknown_5:�@
	unknown_6:@ 
	unknown_7:�@
	unknown_8:@ 
	unknown_9:@�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�

unknown_18:	�!

unknown_19:�@

unknown_20:@!

unknown_21:�@

unknown_22:@!

unknown_23:�@

unknown_24:@!

unknown_25:@�

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:
��

unknown_30:	�

unknown_31:
��

unknown_32:	�

unknown_33:	�

unknown_34:	�!

unknown_35:�@

unknown_36:@!

unknown_37:�@

unknown_38:@!

unknown_39:�@

unknown_40:@!

unknown_41:@�

unknown_42:	�

unknown_43:	�

unknown_44:	�

unknown_45:
��

unknown_46:	�

unknown_47:
��

unknown_48:	�

unknown_49:	�

unknown_50:	�!

unknown_51:�@

unknown_52:@!

unknown_53:�@

unknown_54:@!

unknown_55:�@

unknown_56:@!

unknown_57:@�

unknown_58:	�

unknown_59:	�

unknown_60:	�

unknown_61:
��

unknown_62:	�

unknown_63:
��

unknown_64:	�

unknown_65:	�

unknown_66:	�!

unknown_67:�@

unknown_68:@!

unknown_69:�@

unknown_70:@!

unknown_71:�@

unknown_72:@!

unknown_73:@�

unknown_74:	�

unknown_75:	�

unknown_76:	�

unknown_77:
��

unknown_78:	�

unknown_79:
��

unknown_80:	�

unknown_81:	�

unknown_82:	�!

unknown_83:�@

unknown_84:@!

unknown_85:�@

unknown_86:@!

unknown_87:�@

unknown_88:@!

unknown_89:@�

unknown_90:	�

unknown_91:	�

unknown_92:	�

unknown_93:
��

unknown_94:	�

unknown_95:
��

unknown_96:	�

unknown_97:	�

unknown_98:	�

unknown_99:
��
unknown_100:	�
unknown_101:	�
unknown_102:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallpadding_mask	token_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69
unknown_70
unknown_71
unknown_72
unknown_73
unknown_74
unknown_75
unknown_76
unknown_77
unknown_78
unknown_79
unknown_80
unknown_81
unknown_82
unknown_83
unknown_84
unknown_85
unknown_86
unknown_87
unknown_88
unknown_89
unknown_90
unknown_91
unknown_92
unknown_93
unknown_94
unknown_95
unknown_96
unknown_97
unknown_98
unknown_99unknown_100unknown_101unknown_102*u
Tinn
l2j*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*�
_read_only_resource_inputsl
jh	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghi*-
config_proto

CPU

GPU 2J 8� *"
fR
__inference___call___8096o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:������������������:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:������������������
&
_user_specified_namepadding_mask:[W
0
_output_shapes
:������������������
#
_user_specified_name	token_ids:$ 

_user_specified_name8315:$ 

_user_specified_name8317:$ 

_user_specified_name8319:$ 

_user_specified_name8321:$ 

_user_specified_name8323:$ 

_user_specified_name8325:$ 

_user_specified_name8327:$	 

_user_specified_name8329:$
 

_user_specified_name8331:$ 

_user_specified_name8333:$ 

_user_specified_name8335:$ 

_user_specified_name8337:$ 

_user_specified_name8339:$ 

_user_specified_name8341:$ 

_user_specified_name8343:$ 

_user_specified_name8345:$ 

_user_specified_name8347:$ 

_user_specified_name8349:$ 

_user_specified_name8351:$ 

_user_specified_name8353:$ 

_user_specified_name8355:$ 

_user_specified_name8357:$ 

_user_specified_name8359:$ 

_user_specified_name8361:$ 

_user_specified_name8363:$ 

_user_specified_name8365:$ 

_user_specified_name8367:$ 

_user_specified_name8369:$ 

_user_specified_name8371:$ 

_user_specified_name8373:$  

_user_specified_name8375:$! 

_user_specified_name8377:$" 

_user_specified_name8379:$# 

_user_specified_name8381:$$ 

_user_specified_name8383:$% 

_user_specified_name8385:$& 

_user_specified_name8387:$' 

_user_specified_name8389:$( 

_user_specified_name8391:$) 

_user_specified_name8393:$* 

_user_specified_name8395:$+ 

_user_specified_name8397:$, 

_user_specified_name8399:$- 

_user_specified_name8401:$. 

_user_specified_name8403:$/ 

_user_specified_name8405:$0 

_user_specified_name8407:$1 

_user_specified_name8409:$2 

_user_specified_name8411:$3 

_user_specified_name8413:$4 

_user_specified_name8415:$5 

_user_specified_name8417:$6 

_user_specified_name8419:$7 

_user_specified_name8421:$8 

_user_specified_name8423:$9 

_user_specified_name8425:$: 

_user_specified_name8427:$; 

_user_specified_name8429:$< 

_user_specified_name8431:$= 

_user_specified_name8433:$> 

_user_specified_name8435:$? 

_user_specified_name8437:$@ 

_user_specified_name8439:$A 

_user_specified_name8441:$B 

_user_specified_name8443:$C 

_user_specified_name8445:$D 

_user_specified_name8447:$E 

_user_specified_name8449:$F 

_user_specified_name8451:$G 

_user_specified_name8453:$H 

_user_specified_name8455:$I 

_user_specified_name8457:$J 

_user_specified_name8459:$K 

_user_specified_name8461:$L 

_user_specified_name8463:$M 

_user_specified_name8465:$N 

_user_specified_name8467:$O 

_user_specified_name8469:$P 

_user_specified_name8471:$Q 

_user_specified_name8473:$R 

_user_specified_name8475:$S 

_user_specified_name8477:$T 

_user_specified_name8479:$U 

_user_specified_name8481:$V 

_user_specified_name8483:$W 

_user_specified_name8485:$X 

_user_specified_name8487:$Y 

_user_specified_name8489:$Z 

_user_specified_name8491:$[ 

_user_specified_name8493:$\ 

_user_specified_name8495:$] 

_user_specified_name8497:$^ 

_user_specified_name8499:$_ 

_user_specified_name8501:$` 

_user_specified_name8503:$a 

_user_specified_name8505:$b 

_user_specified_name8507:$c 

_user_specified_name8509:$d 

_user_specified_name8511:$e 

_user_specified_name8513:$f 

_user_specified_name8515:$g 

_user_specified_name8517:$h 

_user_specified_name8519:$i 

_user_specified_name8521
��
��
__inference__traced_save_10415
file_prefix8
#read_disablecopyonread_variable_123:���9
%read_1_disablecopyonread_variable_122:
��4
%read_2_disablecopyonread_variable_121:	�4
%read_3_disablecopyonread_variable_120:	�3
%read_4_disablecopyonread_variable_119:<
%read_5_disablecopyonread_variable_118:�@7
%read_6_disablecopyonread_variable_117:@<
%read_7_disablecopyonread_variable_116:�@7
%read_8_disablecopyonread_variable_115:@<
%read_9_disablecopyonread_variable_114:�@8
&read_10_disablecopyonread_variable_113:@4
&read_11_disablecopyonread_variable_112:=
&read_12_disablecopyonread_variable_111:@�5
&read_13_disablecopyonread_variable_110:	�5
&read_14_disablecopyonread_variable_109:	�5
&read_15_disablecopyonread_variable_108:	�4
&read_16_disablecopyonread_variable_107:5
&read_17_disablecopyonread_variable_106:	�5
&read_18_disablecopyonread_variable_105:	�:
&read_19_disablecopyonread_variable_104:
��5
&read_20_disablecopyonread_variable_103:	�:
&read_21_disablecopyonread_variable_102:
��5
&read_22_disablecopyonread_variable_101:	�4
&read_23_disablecopyonread_variable_100:<
%read_24_disablecopyonread_variable_99:�@7
%read_25_disablecopyonread_variable_98:@<
%read_26_disablecopyonread_variable_97:�@7
%read_27_disablecopyonread_variable_96:@<
%read_28_disablecopyonread_variable_95:�@7
%read_29_disablecopyonread_variable_94:@3
%read_30_disablecopyonread_variable_93:<
%read_31_disablecopyonread_variable_92:@�4
%read_32_disablecopyonread_variable_91:	�4
%read_33_disablecopyonread_variable_90:	�4
%read_34_disablecopyonread_variable_89:	�3
%read_35_disablecopyonread_variable_88:4
%read_36_disablecopyonread_variable_87:	�4
%read_37_disablecopyonread_variable_86:	�9
%read_38_disablecopyonread_variable_85:
��4
%read_39_disablecopyonread_variable_84:	�9
%read_40_disablecopyonread_variable_83:
��4
%read_41_disablecopyonread_variable_82:	�3
%read_42_disablecopyonread_variable_81:<
%read_43_disablecopyonread_variable_80:�@7
%read_44_disablecopyonread_variable_79:@<
%read_45_disablecopyonread_variable_78:�@7
%read_46_disablecopyonread_variable_77:@<
%read_47_disablecopyonread_variable_76:�@7
%read_48_disablecopyonread_variable_75:@3
%read_49_disablecopyonread_variable_74:<
%read_50_disablecopyonread_variable_73:@�4
%read_51_disablecopyonread_variable_72:	�4
%read_52_disablecopyonread_variable_71:	�4
%read_53_disablecopyonread_variable_70:	�3
%read_54_disablecopyonread_variable_69:4
%read_55_disablecopyonread_variable_68:	�4
%read_56_disablecopyonread_variable_67:	�9
%read_57_disablecopyonread_variable_66:
��4
%read_58_disablecopyonread_variable_65:	�9
%read_59_disablecopyonread_variable_64:
��4
%read_60_disablecopyonread_variable_63:	�3
%read_61_disablecopyonread_variable_62:<
%read_62_disablecopyonread_variable_61:�@7
%read_63_disablecopyonread_variable_60:@<
%read_64_disablecopyonread_variable_59:�@7
%read_65_disablecopyonread_variable_58:@<
%read_66_disablecopyonread_variable_57:�@7
%read_67_disablecopyonread_variable_56:@3
%read_68_disablecopyonread_variable_55:<
%read_69_disablecopyonread_variable_54:@�4
%read_70_disablecopyonread_variable_53:	�4
%read_71_disablecopyonread_variable_52:	�4
%read_72_disablecopyonread_variable_51:	�3
%read_73_disablecopyonread_variable_50:4
%read_74_disablecopyonread_variable_49:	�4
%read_75_disablecopyonread_variable_48:	�9
%read_76_disablecopyonread_variable_47:
��4
%read_77_disablecopyonread_variable_46:	�9
%read_78_disablecopyonread_variable_45:
��4
%read_79_disablecopyonread_variable_44:	�3
%read_80_disablecopyonread_variable_43:<
%read_81_disablecopyonread_variable_42:�@7
%read_82_disablecopyonread_variable_41:@<
%read_83_disablecopyonread_variable_40:�@7
%read_84_disablecopyonread_variable_39:@<
%read_85_disablecopyonread_variable_38:�@7
%read_86_disablecopyonread_variable_37:@3
%read_87_disablecopyonread_variable_36:<
%read_88_disablecopyonread_variable_35:@�4
%read_89_disablecopyonread_variable_34:	�4
%read_90_disablecopyonread_variable_33:	�4
%read_91_disablecopyonread_variable_32:	�3
%read_92_disablecopyonread_variable_31:4
%read_93_disablecopyonread_variable_30:	�4
%read_94_disablecopyonread_variable_29:	�9
%read_95_disablecopyonread_variable_28:
��4
%read_96_disablecopyonread_variable_27:	�9
%read_97_disablecopyonread_variable_26:
��4
%read_98_disablecopyonread_variable_25:	�3
%read_99_disablecopyonread_variable_24:=
&read_100_disablecopyonread_variable_23:�@8
&read_101_disablecopyonread_variable_22:@=
&read_102_disablecopyonread_variable_21:�@8
&read_103_disablecopyonread_variable_20:@=
&read_104_disablecopyonread_variable_19:�@8
&read_105_disablecopyonread_variable_18:@4
&read_106_disablecopyonread_variable_17:=
&read_107_disablecopyonread_variable_16:@�5
&read_108_disablecopyonread_variable_15:	�5
&read_109_disablecopyonread_variable_14:	�5
&read_110_disablecopyonread_variable_13:	�4
&read_111_disablecopyonread_variable_12:5
&read_112_disablecopyonread_variable_11:	�5
&read_113_disablecopyonread_variable_10:	�9
%read_114_disablecopyonread_variable_9:
��4
%read_115_disablecopyonread_variable_8:	�9
%read_116_disablecopyonread_variable_7:
��4
%read_117_disablecopyonread_variable_6:	�3
%read_118_disablecopyonread_variable_5:9
%read_119_disablecopyonread_variable_4:
��4
%read_120_disablecopyonread_variable_3:	�3
%read_121_disablecopyonread_variable_2:8
%read_122_disablecopyonread_variable_1:	�1
#read_123_disablecopyonread_variable:i
Rread_124_disablecopyonread_transformer_layer_3_self_attention_layer_query_kernel_1:�@b
Pread_125_disablecopyonread_transformer_layer_3_self_attention_layer_value_bias_1:@j
[read_126_disablecopyonread_transformer_layer_3_self_attention_layer_attention_output_bias_1:	�]
Nread_127_disablecopyonread_transformer_layer_3_feedforward_output_dense_bias_1:	�^
Oread_128_disablecopyonread_transformer_layer_5_self_attention_layer_norm_beta_1:	�b
Pread_129_disablecopyonread_transformer_layer_1_self_attention_layer_query_bias_1:@g
Pread_130_disablecopyonread_transformer_layer_3_self_attention_layer_key_kernel_1:�@^
Oread_131_disablecopyonread_transformer_layer_3_self_attention_layer_norm_beta_1:	�`
Nread_132_disablecopyonread_transformer_layer_1_self_attention_layer_key_bias_1:@]
Nread_133_disablecopyonread_transformer_layer_4_feedforward_output_dense_bias_1:	�\
Mread_134_disablecopyonread_transformer_layer_4_feedforward_layer_norm_gamma_1:	�[
Lread_135_disablecopyonread_transformer_layer_0_feedforward_layer_norm_beta_1:	�b
Pread_136_disablecopyonread_transformer_layer_4_self_attention_layer_query_bias_1:@g
Pread_137_disablecopyonread_transformer_layer_4_self_attention_layer_key_kernel_1:�@i
Tread_138_disablecopyonread_token_and_position_embedding_token_embedding_embeddings_1:���]
Nread_139_disablecopyonread_transformer_layer_5_feedforward_output_dense_bias_1:	�b
Pread_140_disablecopyonread_transformer_layer_3_self_attention_layer_query_bias_1:@`
Nread_141_disablecopyonread_transformer_layer_3_self_attention_layer_key_bias_1:@_
Pread_142_disablecopyonread_transformer_layer_0_self_attention_layer_norm_gamma_1:	�[
Lread_143_disablecopyonread_transformer_layer_4_feedforward_layer_norm_beta_1:	�F
7read_144_disablecopyonread_embeddings_layer_norm_beta_1:	�i
Rread_145_disablecopyonread_transformer_layer_2_self_attention_layer_query_kernel_1:�@\
Mread_146_disablecopyonread_transformer_layer_2_feedforward_layer_norm_gamma_1:	�c
Tread_147_disablecopyonread_transformer_layer_4_feedforward_intermediate_dense_bias_1:	�`
Nread_148_disablecopyonread_transformer_layer_4_self_attention_layer_key_bias_1:@i
Rread_149_disablecopyonread_transformer_layer_4_self_attention_layer_value_kernel_1:�@d
Pread_150_disablecopyonread_transformer_layer_1_feedforward_output_dense_kernel_1:
��t
]read_151_disablecopyonread_transformer_layer_0_self_attention_layer_attention_output_kernel_1:@�\
Mread_152_disablecopyonread_transformer_layer_1_feedforward_layer_norm_gamma_1:	�c
Tread_153_disablecopyonread_transformer_layer_1_feedforward_intermediate_dense_bias_1:	�=
.read_154_disablecopyonread_pooled_dense_bias_1:	�b
Pread_155_disablecopyonread_transformer_layer_2_self_attention_layer_query_bias_1:@[
Lread_156_disablecopyonread_transformer_layer_2_feedforward_layer_norm_beta_1:	�j
Vread_157_disablecopyonread_transformer_layer_4_feedforward_intermediate_dense_kernel_1:
��t
]read_158_disablecopyonread_transformer_layer_4_self_attention_layer_attention_output_kernel_1:@�k
Wread_159_disablecopyonread_token_and_position_embedding_position_embedding_embeddings_1:
��g
Pread_160_disablecopyonread_transformer_layer_2_self_attention_layer_key_kernel_1:�@c
Tread_161_disablecopyonread_transformer_layer_2_feedforward_intermediate_dense_bias_1:	�b
Pread_162_disablecopyonread_transformer_layer_4_self_attention_layer_value_bias_1:@\
Mread_163_disablecopyonread_transformer_layer_5_feedforward_layer_norm_gamma_1:	�=
*read_164_disablecopyonread_logits_kernel_1:	�G
8read_165_disablecopyonread_embeddings_layer_norm_gamma_1:	�i
Rread_166_disablecopyonread_transformer_layer_2_self_attention_layer_value_kernel_1:�@\
Mread_167_disablecopyonread_transformer_layer_3_feedforward_layer_norm_gamma_1:	�_
Pread_168_disablecopyonread_transformer_layer_4_self_attention_layer_norm_gamma_1:	�i
Rread_169_disablecopyonread_transformer_layer_5_self_attention_layer_query_kernel_1:�@i
Rread_170_disablecopyonread_transformer_layer_0_self_attention_layer_query_kernel_1:�@j
[read_171_disablecopyonread_transformer_layer_0_self_attention_layer_attention_output_bias_1:	�[
Lread_172_disablecopyonread_transformer_layer_1_feedforward_layer_norm_beta_1:	�g
Pread_173_disablecopyonread_transformer_layer_5_self_attention_layer_key_kernel_1:�@g
Pread_174_disablecopyonread_transformer_layer_0_self_attention_layer_key_kernel_1:�@^
Oread_175_disablecopyonread_transformer_layer_0_self_attention_layer_norm_beta_1:	�i
Rread_176_disablecopyonread_transformer_layer_5_self_attention_layer_value_kernel_1:�@i
Rread_177_disablecopyonread_transformer_layer_1_self_attention_layer_value_kernel_1:�@`
Nread_178_disablecopyonread_transformer_layer_2_self_attention_layer_key_bias_1:@j
Vread_179_disablecopyonread_transformer_layer_2_feedforward_intermediate_dense_kernel_1:
��j
[read_180_disablecopyonread_transformer_layer_4_self_attention_layer_attention_output_bias_1:	�[
Lread_181_disablecopyonread_transformer_layer_5_feedforward_layer_norm_beta_1:	�D
0read_182_disablecopyonread_pooled_dense_kernel_1:
��6
(read_183_disablecopyonread_logits_bias_1:]
Nread_184_disablecopyonread_transformer_layer_0_feedforward_output_dense_bias_1:	�b
Pread_185_disablecopyonread_transformer_layer_2_self_attention_layer_value_bias_1:@t
]read_186_disablecopyonread_transformer_layer_2_self_attention_layer_attention_output_kernel_1:@�[
Lread_187_disablecopyonread_transformer_layer_3_feedforward_layer_norm_beta_1:	�^
Oread_188_disablecopyonread_transformer_layer_4_self_attention_layer_norm_beta_1:	�b
Pread_189_disablecopyonread_transformer_layer_5_self_attention_layer_query_bias_1:@j
Vread_190_disablecopyonread_transformer_layer_5_feedforward_intermediate_dense_kernel_1:
��b
Pread_191_disablecopyonread_transformer_layer_0_self_attention_layer_query_bias_1:@_
Pread_192_disablecopyonread_transformer_layer_2_self_attention_layer_norm_gamma_1:	�j
Vread_193_disablecopyonread_transformer_layer_3_feedforward_intermediate_dense_kernel_1:
��`
Nread_194_disablecopyonread_transformer_layer_0_self_attention_layer_key_bias_1:@j
Vread_195_disablecopyonread_transformer_layer_1_feedforward_intermediate_dense_kernel_1:
��b
Pread_196_disablecopyonread_transformer_layer_5_self_attention_layer_value_bias_1:@t
]read_197_disablecopyonread_transformer_layer_5_self_attention_layer_attention_output_kernel_1:@�i
Rread_198_disablecopyonread_transformer_layer_0_self_attention_layer_value_kernel_1:�@b
Pread_199_disablecopyonread_transformer_layer_1_self_attention_layer_value_bias_1:@t
]read_200_disablecopyonread_transformer_layer_1_self_attention_layer_attention_output_kernel_1:@�j
Vread_201_disablecopyonread_transformer_layer_0_feedforward_intermediate_dense_kernel_1:
��_
Pread_202_disablecopyonread_transformer_layer_1_self_attention_layer_norm_gamma_1:	�j
[read_203_disablecopyonread_transformer_layer_2_self_attention_layer_attention_output_bias_1:	�d
Pread_204_disablecopyonread_transformer_layer_2_feedforward_output_dense_kernel_1:
��c
Tread_205_disablecopyonread_transformer_layer_5_feedforward_intermediate_dense_bias_1:	�^
Oread_206_disablecopyonread_transformer_layer_2_self_attention_layer_norm_beta_1:	�c
Tread_207_disablecopyonread_transformer_layer_3_feedforward_intermediate_dense_bias_1:	�`
Nread_208_disablecopyonread_transformer_layer_5_self_attention_layer_key_bias_1:@]
Nread_209_disablecopyonread_transformer_layer_1_feedforward_output_dense_bias_1:	�i
Rread_210_disablecopyonread_transformer_layer_3_self_attention_layer_value_kernel_1:�@t
]read_211_disablecopyonread_transformer_layer_3_self_attention_layer_attention_output_kernel_1:@�d
Pread_212_disablecopyonread_transformer_layer_3_feedforward_output_dense_kernel_1:
��j
[read_213_disablecopyonread_transformer_layer_5_self_attention_layer_attention_output_bias_1:	�_
Pread_214_disablecopyonread_transformer_layer_5_self_attention_layer_norm_gamma_1:	�b
Pread_215_disablecopyonread_transformer_layer_0_self_attention_layer_value_bias_1:@i
Rread_216_disablecopyonread_transformer_layer_1_self_attention_layer_query_kernel_1:�@j
[read_217_disablecopyonread_transformer_layer_1_self_attention_layer_attention_output_bias_1:	�_
Pread_218_disablecopyonread_transformer_layer_3_self_attention_layer_norm_gamma_1:	�c
Tread_219_disablecopyonread_transformer_layer_0_feedforward_intermediate_dense_bias_1:	�g
Pread_220_disablecopyonread_transformer_layer_1_self_attention_layer_key_kernel_1:�@^
Oread_221_disablecopyonread_transformer_layer_1_self_attention_layer_norm_beta_1:	�d
Pread_222_disablecopyonread_transformer_layer_4_feedforward_output_dense_kernel_1:
��i
Rread_223_disablecopyonread_transformer_layer_4_self_attention_layer_query_kernel_1:�@d
Pread_224_disablecopyonread_transformer_layer_0_feedforward_output_dense_kernel_1:
��\
Mread_225_disablecopyonread_transformer_layer_0_feedforward_layer_norm_gamma_1:	�]
Nread_226_disablecopyonread_transformer_layer_2_feedforward_output_dense_bias_1:	�d
Pread_227_disablecopyonread_transformer_layer_5_feedforward_output_dense_kernel_1:
��
savev2_const
identity_457��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_100/DisableCopyOnRead�Read_100/ReadVariableOp�Read_101/DisableCopyOnRead�Read_101/ReadVariableOp�Read_102/DisableCopyOnRead�Read_102/ReadVariableOp�Read_103/DisableCopyOnRead�Read_103/ReadVariableOp�Read_104/DisableCopyOnRead�Read_104/ReadVariableOp�Read_105/DisableCopyOnRead�Read_105/ReadVariableOp�Read_106/DisableCopyOnRead�Read_106/ReadVariableOp�Read_107/DisableCopyOnRead�Read_107/ReadVariableOp�Read_108/DisableCopyOnRead�Read_108/ReadVariableOp�Read_109/DisableCopyOnRead�Read_109/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_110/DisableCopyOnRead�Read_110/ReadVariableOp�Read_111/DisableCopyOnRead�Read_111/ReadVariableOp�Read_112/DisableCopyOnRead�Read_112/ReadVariableOp�Read_113/DisableCopyOnRead�Read_113/ReadVariableOp�Read_114/DisableCopyOnRead�Read_114/ReadVariableOp�Read_115/DisableCopyOnRead�Read_115/ReadVariableOp�Read_116/DisableCopyOnRead�Read_116/ReadVariableOp�Read_117/DisableCopyOnRead�Read_117/ReadVariableOp�Read_118/DisableCopyOnRead�Read_118/ReadVariableOp�Read_119/DisableCopyOnRead�Read_119/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_120/DisableCopyOnRead�Read_120/ReadVariableOp�Read_121/DisableCopyOnRead�Read_121/ReadVariableOp�Read_122/DisableCopyOnRead�Read_122/ReadVariableOp�Read_123/DisableCopyOnRead�Read_123/ReadVariableOp�Read_124/DisableCopyOnRead�Read_124/ReadVariableOp�Read_125/DisableCopyOnRead�Read_125/ReadVariableOp�Read_126/DisableCopyOnRead�Read_126/ReadVariableOp�Read_127/DisableCopyOnRead�Read_127/ReadVariableOp�Read_128/DisableCopyOnRead�Read_128/ReadVariableOp�Read_129/DisableCopyOnRead�Read_129/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_130/DisableCopyOnRead�Read_130/ReadVariableOp�Read_131/DisableCopyOnRead�Read_131/ReadVariableOp�Read_132/DisableCopyOnRead�Read_132/ReadVariableOp�Read_133/DisableCopyOnRead�Read_133/ReadVariableOp�Read_134/DisableCopyOnRead�Read_134/ReadVariableOp�Read_135/DisableCopyOnRead�Read_135/ReadVariableOp�Read_136/DisableCopyOnRead�Read_136/ReadVariableOp�Read_137/DisableCopyOnRead�Read_137/ReadVariableOp�Read_138/DisableCopyOnRead�Read_138/ReadVariableOp�Read_139/DisableCopyOnRead�Read_139/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_140/DisableCopyOnRead�Read_140/ReadVariableOp�Read_141/DisableCopyOnRead�Read_141/ReadVariableOp�Read_142/DisableCopyOnRead�Read_142/ReadVariableOp�Read_143/DisableCopyOnRead�Read_143/ReadVariableOp�Read_144/DisableCopyOnRead�Read_144/ReadVariableOp�Read_145/DisableCopyOnRead�Read_145/ReadVariableOp�Read_146/DisableCopyOnRead�Read_146/ReadVariableOp�Read_147/DisableCopyOnRead�Read_147/ReadVariableOp�Read_148/DisableCopyOnRead�Read_148/ReadVariableOp�Read_149/DisableCopyOnRead�Read_149/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_150/DisableCopyOnRead�Read_150/ReadVariableOp�Read_151/DisableCopyOnRead�Read_151/ReadVariableOp�Read_152/DisableCopyOnRead�Read_152/ReadVariableOp�Read_153/DisableCopyOnRead�Read_153/ReadVariableOp�Read_154/DisableCopyOnRead�Read_154/ReadVariableOp�Read_155/DisableCopyOnRead�Read_155/ReadVariableOp�Read_156/DisableCopyOnRead�Read_156/ReadVariableOp�Read_157/DisableCopyOnRead�Read_157/ReadVariableOp�Read_158/DisableCopyOnRead�Read_158/ReadVariableOp�Read_159/DisableCopyOnRead�Read_159/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_160/DisableCopyOnRead�Read_160/ReadVariableOp�Read_161/DisableCopyOnRead�Read_161/ReadVariableOp�Read_162/DisableCopyOnRead�Read_162/ReadVariableOp�Read_163/DisableCopyOnRead�Read_163/ReadVariableOp�Read_164/DisableCopyOnRead�Read_164/ReadVariableOp�Read_165/DisableCopyOnRead�Read_165/ReadVariableOp�Read_166/DisableCopyOnRead�Read_166/ReadVariableOp�Read_167/DisableCopyOnRead�Read_167/ReadVariableOp�Read_168/DisableCopyOnRead�Read_168/ReadVariableOp�Read_169/DisableCopyOnRead�Read_169/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_170/DisableCopyOnRead�Read_170/ReadVariableOp�Read_171/DisableCopyOnRead�Read_171/ReadVariableOp�Read_172/DisableCopyOnRead�Read_172/ReadVariableOp�Read_173/DisableCopyOnRead�Read_173/ReadVariableOp�Read_174/DisableCopyOnRead�Read_174/ReadVariableOp�Read_175/DisableCopyOnRead�Read_175/ReadVariableOp�Read_176/DisableCopyOnRead�Read_176/ReadVariableOp�Read_177/DisableCopyOnRead�Read_177/ReadVariableOp�Read_178/DisableCopyOnRead�Read_178/ReadVariableOp�Read_179/DisableCopyOnRead�Read_179/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_180/DisableCopyOnRead�Read_180/ReadVariableOp�Read_181/DisableCopyOnRead�Read_181/ReadVariableOp�Read_182/DisableCopyOnRead�Read_182/ReadVariableOp�Read_183/DisableCopyOnRead�Read_183/ReadVariableOp�Read_184/DisableCopyOnRead�Read_184/ReadVariableOp�Read_185/DisableCopyOnRead�Read_185/ReadVariableOp�Read_186/DisableCopyOnRead�Read_186/ReadVariableOp�Read_187/DisableCopyOnRead�Read_187/ReadVariableOp�Read_188/DisableCopyOnRead�Read_188/ReadVariableOp�Read_189/DisableCopyOnRead�Read_189/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_190/DisableCopyOnRead�Read_190/ReadVariableOp�Read_191/DisableCopyOnRead�Read_191/ReadVariableOp�Read_192/DisableCopyOnRead�Read_192/ReadVariableOp�Read_193/DisableCopyOnRead�Read_193/ReadVariableOp�Read_194/DisableCopyOnRead�Read_194/ReadVariableOp�Read_195/DisableCopyOnRead�Read_195/ReadVariableOp�Read_196/DisableCopyOnRead�Read_196/ReadVariableOp�Read_197/DisableCopyOnRead�Read_197/ReadVariableOp�Read_198/DisableCopyOnRead�Read_198/ReadVariableOp�Read_199/DisableCopyOnRead�Read_199/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_200/DisableCopyOnRead�Read_200/ReadVariableOp�Read_201/DisableCopyOnRead�Read_201/ReadVariableOp�Read_202/DisableCopyOnRead�Read_202/ReadVariableOp�Read_203/DisableCopyOnRead�Read_203/ReadVariableOp�Read_204/DisableCopyOnRead�Read_204/ReadVariableOp�Read_205/DisableCopyOnRead�Read_205/ReadVariableOp�Read_206/DisableCopyOnRead�Read_206/ReadVariableOp�Read_207/DisableCopyOnRead�Read_207/ReadVariableOp�Read_208/DisableCopyOnRead�Read_208/ReadVariableOp�Read_209/DisableCopyOnRead�Read_209/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_210/DisableCopyOnRead�Read_210/ReadVariableOp�Read_211/DisableCopyOnRead�Read_211/ReadVariableOp�Read_212/DisableCopyOnRead�Read_212/ReadVariableOp�Read_213/DisableCopyOnRead�Read_213/ReadVariableOp�Read_214/DisableCopyOnRead�Read_214/ReadVariableOp�Read_215/DisableCopyOnRead�Read_215/ReadVariableOp�Read_216/DisableCopyOnRead�Read_216/ReadVariableOp�Read_217/DisableCopyOnRead�Read_217/ReadVariableOp�Read_218/DisableCopyOnRead�Read_218/ReadVariableOp�Read_219/DisableCopyOnRead�Read_219/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_220/DisableCopyOnRead�Read_220/ReadVariableOp�Read_221/DisableCopyOnRead�Read_221/ReadVariableOp�Read_222/DisableCopyOnRead�Read_222/ReadVariableOp�Read_223/DisableCopyOnRead�Read_223/ReadVariableOp�Read_224/DisableCopyOnRead�Read_224/ReadVariableOp�Read_225/DisableCopyOnRead�Read_225/ReadVariableOp�Read_226/DisableCopyOnRead�Read_226/ReadVariableOp�Read_227/DisableCopyOnRead�Read_227/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_88/DisableCopyOnRead�Read_88/ReadVariableOp�Read_89/DisableCopyOnRead�Read_89/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOp�Read_90/DisableCopyOnRead�Read_90/ReadVariableOp�Read_91/DisableCopyOnRead�Read_91/ReadVariableOp�Read_92/DisableCopyOnRead�Read_92/ReadVariableOp�Read_93/DisableCopyOnRead�Read_93/ReadVariableOp�Read_94/DisableCopyOnRead�Read_94/ReadVariableOp�Read_95/DisableCopyOnRead�Read_95/ReadVariableOp�Read_96/DisableCopyOnRead�Read_96/ReadVariableOp�Read_97/DisableCopyOnRead�Read_97/ReadVariableOp�Read_98/DisableCopyOnRead�Read_98/ReadVariableOp�Read_99/DisableCopyOnRead�Read_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: f
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_variable_123*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_variable_123^Read/DisableCopyOnRead*!
_output_shapes
:���*
dtype0]
IdentityIdentityRead/ReadVariableOp:value:0*
T0*!
_output_shapes
:���d

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*!
_output_shapes
:���j
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_variable_122*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_variable_122^Read_1/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0`

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��e

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��j
Read_2/DisableCopyOnReadDisableCopyOnRead%read_2_disablecopyonread_variable_121*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp%read_2_disablecopyonread_variable_121^Read_2/DisableCopyOnRead*
_output_shapes	
:�*
dtype0[

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_variable_120*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_variable_120^Read_3/DisableCopyOnRead*
_output_shapes	
:�*
dtype0[

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_4/DisableCopyOnReadDisableCopyOnRead%read_4_disablecopyonread_variable_119*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp%read_4_disablecopyonread_variable_119^Read_4/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:j
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_variable_118*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_variable_118^Read_5/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0d
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@j
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@j
Read_6/DisableCopyOnReadDisableCopyOnRead%read_6_disablecopyonread_variable_117*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp%read_6_disablecopyonread_variable_117^Read_6/DisableCopyOnRead*
_output_shapes

:@*
dtype0_
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:@j
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_variable_116*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_variable_116^Read_7/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0d
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@j
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@j
Read_8/DisableCopyOnReadDisableCopyOnRead%read_8_disablecopyonread_variable_115*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp%read_8_disablecopyonread_variable_115^Read_8/DisableCopyOnRead*
_output_shapes

:@*
dtype0_
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:@j
Read_9/DisableCopyOnReadDisableCopyOnRead%read_9_disablecopyonread_variable_114*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp%read_9_disablecopyonread_variable_114^Read_9/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0d
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@j
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@l
Read_10/DisableCopyOnReadDisableCopyOnRead&read_10_disablecopyonread_variable_113*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp&read_10_disablecopyonread_variable_113^Read_10/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:@l
Read_11/DisableCopyOnReadDisableCopyOnRead&read_11_disablecopyonread_variable_112*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp&read_11_disablecopyonread_variable_112^Read_11/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:l
Read_12/DisableCopyOnReadDisableCopyOnRead&read_12_disablecopyonread_variable_111*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp&read_12_disablecopyonread_variable_111^Read_12/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0e
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�j
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*#
_output_shapes
:@�l
Read_13/DisableCopyOnReadDisableCopyOnRead&read_13_disablecopyonread_variable_110*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp&read_13_disablecopyonread_variable_110^Read_13/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_14/DisableCopyOnReadDisableCopyOnRead&read_14_disablecopyonread_variable_109*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp&read_14_disablecopyonread_variable_109^Read_14/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_15/DisableCopyOnReadDisableCopyOnRead&read_15_disablecopyonread_variable_108*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp&read_15_disablecopyonread_variable_108^Read_15/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_16/DisableCopyOnReadDisableCopyOnRead&read_16_disablecopyonread_variable_107*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp&read_16_disablecopyonread_variable_107^Read_16/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:l
Read_17/DisableCopyOnReadDisableCopyOnRead&read_17_disablecopyonread_variable_106*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp&read_17_disablecopyonread_variable_106^Read_17/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_18/DisableCopyOnReadDisableCopyOnRead&read_18_disablecopyonread_variable_105*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp&read_18_disablecopyonread_variable_105^Read_18/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_19/DisableCopyOnReadDisableCopyOnRead&read_19_disablecopyonread_variable_104*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp&read_19_disablecopyonread_variable_104^Read_19/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��l
Read_20/DisableCopyOnReadDisableCopyOnRead&read_20_disablecopyonread_variable_103*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp&read_20_disablecopyonread_variable_103^Read_20/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_21/DisableCopyOnReadDisableCopyOnRead&read_21_disablecopyonread_variable_102*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp&read_21_disablecopyonread_variable_102^Read_21/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��l
Read_22/DisableCopyOnReadDisableCopyOnRead&read_22_disablecopyonread_variable_101*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp&read_22_disablecopyonread_variable_101^Read_22/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_23/DisableCopyOnReadDisableCopyOnRead&read_23_disablecopyonread_variable_100*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp&read_23_disablecopyonread_variable_100^Read_23/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_24/DisableCopyOnReadDisableCopyOnRead%read_24_disablecopyonread_variable_99*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp%read_24_disablecopyonread_variable_99^Read_24/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0e
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@j
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_25/DisableCopyOnReadDisableCopyOnRead%read_25_disablecopyonread_variable_98*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp%read_25_disablecopyonread_variable_98^Read_25/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_26/DisableCopyOnReadDisableCopyOnRead%read_26_disablecopyonread_variable_97*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp%read_26_disablecopyonread_variable_97^Read_26/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0e
Identity_52IdentityRead_26/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@j
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_27/DisableCopyOnReadDisableCopyOnRead%read_27_disablecopyonread_variable_96*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp%read_27_disablecopyonread_variable_96^Read_27/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_54IdentityRead_27/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_28/DisableCopyOnReadDisableCopyOnRead%read_28_disablecopyonread_variable_95*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp%read_28_disablecopyonread_variable_95^Read_28/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0e
Identity_56IdentityRead_28/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@j
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_29/DisableCopyOnReadDisableCopyOnRead%read_29_disablecopyonread_variable_94*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp%read_29_disablecopyonread_variable_94^Read_29/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_58IdentityRead_29/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_30/DisableCopyOnReadDisableCopyOnRead%read_30_disablecopyonread_variable_93*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp%read_30_disablecopyonread_variable_93^Read_30/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_60IdentityRead_30/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_31/DisableCopyOnReadDisableCopyOnRead%read_31_disablecopyonread_variable_92*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp%read_31_disablecopyonread_variable_92^Read_31/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0e
Identity_62IdentityRead_31/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�j
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*#
_output_shapes
:@�k
Read_32/DisableCopyOnReadDisableCopyOnRead%read_32_disablecopyonread_variable_91*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp%read_32_disablecopyonread_variable_91^Read_32/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_64IdentityRead_32/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_33/DisableCopyOnReadDisableCopyOnRead%read_33_disablecopyonread_variable_90*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp%read_33_disablecopyonread_variable_90^Read_33/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_66IdentityRead_33/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_34/DisableCopyOnReadDisableCopyOnRead%read_34_disablecopyonread_variable_89*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp%read_34_disablecopyonread_variable_89^Read_34/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_68IdentityRead_34/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_35/DisableCopyOnReadDisableCopyOnRead%read_35_disablecopyonread_variable_88*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp%read_35_disablecopyonread_variable_88^Read_35/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_70IdentityRead_35/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_36/DisableCopyOnReadDisableCopyOnRead%read_36_disablecopyonread_variable_87*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp%read_36_disablecopyonread_variable_87^Read_36/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_72IdentityRead_36/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_37/DisableCopyOnReadDisableCopyOnRead%read_37_disablecopyonread_variable_86*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp%read_37_disablecopyonread_variable_86^Read_37/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_74IdentityRead_37/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_38/DisableCopyOnReadDisableCopyOnRead%read_38_disablecopyonread_variable_85*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp%read_38_disablecopyonread_variable_85^Read_38/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_76IdentityRead_38/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_39/DisableCopyOnReadDisableCopyOnRead%read_39_disablecopyonread_variable_84*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp%read_39_disablecopyonread_variable_84^Read_39/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_78IdentityRead_39/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_40/DisableCopyOnReadDisableCopyOnRead%read_40_disablecopyonread_variable_83*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp%read_40_disablecopyonread_variable_83^Read_40/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_80IdentityRead_40/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_41/DisableCopyOnReadDisableCopyOnRead%read_41_disablecopyonread_variable_82*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp%read_41_disablecopyonread_variable_82^Read_41/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_82IdentityRead_41/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_42/DisableCopyOnReadDisableCopyOnRead%read_42_disablecopyonread_variable_81*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp%read_42_disablecopyonread_variable_81^Read_42/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_84IdentityRead_42/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_43/DisableCopyOnReadDisableCopyOnRead%read_43_disablecopyonread_variable_80*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp%read_43_disablecopyonread_variable_80^Read_43/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0e
Identity_86IdentityRead_43/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@j
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_44/DisableCopyOnReadDisableCopyOnRead%read_44_disablecopyonread_variable_79*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp%read_44_disablecopyonread_variable_79^Read_44/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_88IdentityRead_44/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_45/DisableCopyOnReadDisableCopyOnRead%read_45_disablecopyonread_variable_78*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp%read_45_disablecopyonread_variable_78^Read_45/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0e
Identity_90IdentityRead_45/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@j
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_46/DisableCopyOnReadDisableCopyOnRead%read_46_disablecopyonread_variable_77*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp%read_46_disablecopyonread_variable_77^Read_46/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_92IdentityRead_46/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_47/DisableCopyOnReadDisableCopyOnRead%read_47_disablecopyonread_variable_76*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp%read_47_disablecopyonread_variable_76^Read_47/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0e
Identity_94IdentityRead_47/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@j
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_48/DisableCopyOnReadDisableCopyOnRead%read_48_disablecopyonread_variable_75*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp%read_48_disablecopyonread_variable_75^Read_48/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_96IdentityRead_48/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_49/DisableCopyOnReadDisableCopyOnRead%read_49_disablecopyonread_variable_74*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp%read_49_disablecopyonread_variable_74^Read_49/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_98IdentityRead_49/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_50/DisableCopyOnReadDisableCopyOnRead%read_50_disablecopyonread_variable_73*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp%read_50_disablecopyonread_variable_73^Read_50/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0f
Identity_100IdentityRead_50/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�l
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*#
_output_shapes
:@�k
Read_51/DisableCopyOnReadDisableCopyOnRead%read_51_disablecopyonread_variable_72*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp%read_51_disablecopyonread_variable_72^Read_51/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_102IdentityRead_51/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_52/DisableCopyOnReadDisableCopyOnRead%read_52_disablecopyonread_variable_71*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp%read_52_disablecopyonread_variable_71^Read_52/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_104IdentityRead_52/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_53/DisableCopyOnReadDisableCopyOnRead%read_53_disablecopyonread_variable_70*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp%read_53_disablecopyonread_variable_70^Read_53/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_106IdentityRead_53/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_54/DisableCopyOnReadDisableCopyOnRead%read_54_disablecopyonread_variable_69*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp%read_54_disablecopyonread_variable_69^Read_54/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_108IdentityRead_54/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_55/DisableCopyOnReadDisableCopyOnRead%read_55_disablecopyonread_variable_68*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp%read_55_disablecopyonread_variable_68^Read_55/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_110IdentityRead_55/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_56/DisableCopyOnReadDisableCopyOnRead%read_56_disablecopyonread_variable_67*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp%read_56_disablecopyonread_variable_67^Read_56/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_112IdentityRead_56/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_57/DisableCopyOnReadDisableCopyOnRead%read_57_disablecopyonread_variable_66*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp%read_57_disablecopyonread_variable_66^Read_57/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_114IdentityRead_57/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_58/DisableCopyOnReadDisableCopyOnRead%read_58_disablecopyonread_variable_65*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp%read_58_disablecopyonread_variable_65^Read_58/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_116IdentityRead_58/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_59/DisableCopyOnReadDisableCopyOnRead%read_59_disablecopyonread_variable_64*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp%read_59_disablecopyonread_variable_64^Read_59/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_118IdentityRead_59/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_60/DisableCopyOnReadDisableCopyOnRead%read_60_disablecopyonread_variable_63*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp%read_60_disablecopyonread_variable_63^Read_60/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_120IdentityRead_60/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_61/DisableCopyOnReadDisableCopyOnRead%read_61_disablecopyonread_variable_62*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp%read_61_disablecopyonread_variable_62^Read_61/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_122IdentityRead_61/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_62/DisableCopyOnReadDisableCopyOnRead%read_62_disablecopyonread_variable_61*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp%read_62_disablecopyonread_variable_61^Read_62/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0f
Identity_124IdentityRead_62/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_63/DisableCopyOnReadDisableCopyOnRead%read_63_disablecopyonread_variable_60*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp%read_63_disablecopyonread_variable_60^Read_63/DisableCopyOnRead*
_output_shapes

:@*
dtype0a
Identity_126IdentityRead_63/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_64/DisableCopyOnReadDisableCopyOnRead%read_64_disablecopyonread_variable_59*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp%read_64_disablecopyonread_variable_59^Read_64/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0f
Identity_128IdentityRead_64/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_65/DisableCopyOnReadDisableCopyOnRead%read_65_disablecopyonread_variable_58*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp%read_65_disablecopyonread_variable_58^Read_65/DisableCopyOnRead*
_output_shapes

:@*
dtype0a
Identity_130IdentityRead_65/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_66/DisableCopyOnReadDisableCopyOnRead%read_66_disablecopyonread_variable_57*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp%read_66_disablecopyonread_variable_57^Read_66/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0f
Identity_132IdentityRead_66/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_67/DisableCopyOnReadDisableCopyOnRead%read_67_disablecopyonread_variable_56*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp%read_67_disablecopyonread_variable_56^Read_67/DisableCopyOnRead*
_output_shapes

:@*
dtype0a
Identity_134IdentityRead_67/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_68/DisableCopyOnReadDisableCopyOnRead%read_68_disablecopyonread_variable_55*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp%read_68_disablecopyonread_variable_55^Read_68/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_136IdentityRead_68/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_69/DisableCopyOnReadDisableCopyOnRead%read_69_disablecopyonread_variable_54*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp%read_69_disablecopyonread_variable_54^Read_69/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0f
Identity_138IdentityRead_69/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�l
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*#
_output_shapes
:@�k
Read_70/DisableCopyOnReadDisableCopyOnRead%read_70_disablecopyonread_variable_53*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp%read_70_disablecopyonread_variable_53^Read_70/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_140IdentityRead_70/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_71/DisableCopyOnReadDisableCopyOnRead%read_71_disablecopyonread_variable_52*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp%read_71_disablecopyonread_variable_52^Read_71/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_142IdentityRead_71/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_72/DisableCopyOnReadDisableCopyOnRead%read_72_disablecopyonread_variable_51*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp%read_72_disablecopyonread_variable_51^Read_72/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_144IdentityRead_72/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_73/DisableCopyOnReadDisableCopyOnRead%read_73_disablecopyonread_variable_50*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp%read_73_disablecopyonread_variable_50^Read_73/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_146IdentityRead_73/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_74/DisableCopyOnReadDisableCopyOnRead%read_74_disablecopyonread_variable_49*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp%read_74_disablecopyonread_variable_49^Read_74/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_148IdentityRead_74/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_75/DisableCopyOnReadDisableCopyOnRead%read_75_disablecopyonread_variable_48*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp%read_75_disablecopyonread_variable_48^Read_75/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_150IdentityRead_75/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_76/DisableCopyOnReadDisableCopyOnRead%read_76_disablecopyonread_variable_47*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp%read_76_disablecopyonread_variable_47^Read_76/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_152IdentityRead_76/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_77/DisableCopyOnReadDisableCopyOnRead%read_77_disablecopyonread_variable_46*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp%read_77_disablecopyonread_variable_46^Read_77/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_154IdentityRead_77/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_78/DisableCopyOnReadDisableCopyOnRead%read_78_disablecopyonread_variable_45*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOp%read_78_disablecopyonread_variable_45^Read_78/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_156IdentityRead_78/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_79/DisableCopyOnReadDisableCopyOnRead%read_79_disablecopyonread_variable_44*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp%read_79_disablecopyonread_variable_44^Read_79/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_158IdentityRead_79/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_80/DisableCopyOnReadDisableCopyOnRead%read_80_disablecopyonread_variable_43*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOp%read_80_disablecopyonread_variable_43^Read_80/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_160IdentityRead_80/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_81/DisableCopyOnReadDisableCopyOnRead%read_81_disablecopyonread_variable_42*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOp%read_81_disablecopyonread_variable_42^Read_81/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0f
Identity_162IdentityRead_81/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_82/DisableCopyOnReadDisableCopyOnRead%read_82_disablecopyonread_variable_41*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOp%read_82_disablecopyonread_variable_41^Read_82/DisableCopyOnRead*
_output_shapes

:@*
dtype0a
Identity_164IdentityRead_82/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_83/DisableCopyOnReadDisableCopyOnRead%read_83_disablecopyonread_variable_40*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOp%read_83_disablecopyonread_variable_40^Read_83/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0f
Identity_166IdentityRead_83/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_84/DisableCopyOnReadDisableCopyOnRead%read_84_disablecopyonread_variable_39*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOp%read_84_disablecopyonread_variable_39^Read_84/DisableCopyOnRead*
_output_shapes

:@*
dtype0a
Identity_168IdentityRead_84/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_85/DisableCopyOnReadDisableCopyOnRead%read_85_disablecopyonread_variable_38*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOp%read_85_disablecopyonread_variable_38^Read_85/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0f
Identity_170IdentityRead_85/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@k
Read_86/DisableCopyOnReadDisableCopyOnRead%read_86_disablecopyonread_variable_37*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOp%read_86_disablecopyonread_variable_37^Read_86/DisableCopyOnRead*
_output_shapes

:@*
dtype0a
Identity_172IdentityRead_86/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_87/DisableCopyOnReadDisableCopyOnRead%read_87_disablecopyonread_variable_36*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOp%read_87_disablecopyonread_variable_36^Read_87/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_174IdentityRead_87/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_88/DisableCopyOnReadDisableCopyOnRead%read_88_disablecopyonread_variable_35*
_output_shapes
 �
Read_88/ReadVariableOpReadVariableOp%read_88_disablecopyonread_variable_35^Read_88/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0f
Identity_176IdentityRead_88/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�l
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*#
_output_shapes
:@�k
Read_89/DisableCopyOnReadDisableCopyOnRead%read_89_disablecopyonread_variable_34*
_output_shapes
 �
Read_89/ReadVariableOpReadVariableOp%read_89_disablecopyonread_variable_34^Read_89/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_178IdentityRead_89/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_90/DisableCopyOnReadDisableCopyOnRead%read_90_disablecopyonread_variable_33*
_output_shapes
 �
Read_90/ReadVariableOpReadVariableOp%read_90_disablecopyonread_variable_33^Read_90/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_180IdentityRead_90/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_91/DisableCopyOnReadDisableCopyOnRead%read_91_disablecopyonread_variable_32*
_output_shapes
 �
Read_91/ReadVariableOpReadVariableOp%read_91_disablecopyonread_variable_32^Read_91/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_182IdentityRead_91/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_92/DisableCopyOnReadDisableCopyOnRead%read_92_disablecopyonread_variable_31*
_output_shapes
 �
Read_92/ReadVariableOpReadVariableOp%read_92_disablecopyonread_variable_31^Read_92/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_184IdentityRead_92/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_93/DisableCopyOnReadDisableCopyOnRead%read_93_disablecopyonread_variable_30*
_output_shapes
 �
Read_93/ReadVariableOpReadVariableOp%read_93_disablecopyonread_variable_30^Read_93/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_186IdentityRead_93/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_94/DisableCopyOnReadDisableCopyOnRead%read_94_disablecopyonread_variable_29*
_output_shapes
 �
Read_94/ReadVariableOpReadVariableOp%read_94_disablecopyonread_variable_29^Read_94/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_188IdentityRead_94/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_95/DisableCopyOnReadDisableCopyOnRead%read_95_disablecopyonread_variable_28*
_output_shapes
 �
Read_95/ReadVariableOpReadVariableOp%read_95_disablecopyonread_variable_28^Read_95/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_190IdentityRead_95/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_96/DisableCopyOnReadDisableCopyOnRead%read_96_disablecopyonread_variable_27*
_output_shapes
 �
Read_96/ReadVariableOpReadVariableOp%read_96_disablecopyonread_variable_27^Read_96/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_192IdentityRead_96/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_97/DisableCopyOnReadDisableCopyOnRead%read_97_disablecopyonread_variable_26*
_output_shapes
 �
Read_97/ReadVariableOpReadVariableOp%read_97_disablecopyonread_variable_26^Read_97/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_194IdentityRead_97/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��k
Read_98/DisableCopyOnReadDisableCopyOnRead%read_98_disablecopyonread_variable_25*
_output_shapes
 �
Read_98/ReadVariableOpReadVariableOp%read_98_disablecopyonread_variable_25^Read_98/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_196IdentityRead_98/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_99/DisableCopyOnReadDisableCopyOnRead%read_99_disablecopyonread_variable_24*
_output_shapes
 �
Read_99/ReadVariableOpReadVariableOp%read_99_disablecopyonread_variable_24^Read_99/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_198IdentityRead_99/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*
_output_shapes
:m
Read_100/DisableCopyOnReadDisableCopyOnRead&read_100_disablecopyonread_variable_23*
_output_shapes
 �
Read_100/ReadVariableOpReadVariableOp&read_100_disablecopyonread_variable_23^Read_100/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_200IdentityRead_100/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@m
Read_101/DisableCopyOnReadDisableCopyOnRead&read_101_disablecopyonread_variable_22*
_output_shapes
 �
Read_101/ReadVariableOpReadVariableOp&read_101_disablecopyonread_variable_22^Read_101/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_202IdentityRead_101/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*
_output_shapes

:@m
Read_102/DisableCopyOnReadDisableCopyOnRead&read_102_disablecopyonread_variable_21*
_output_shapes
 �
Read_102/ReadVariableOpReadVariableOp&read_102_disablecopyonread_variable_21^Read_102/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_204IdentityRead_102/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@m
Read_103/DisableCopyOnReadDisableCopyOnRead&read_103_disablecopyonread_variable_20*
_output_shapes
 �
Read_103/ReadVariableOpReadVariableOp&read_103_disablecopyonread_variable_20^Read_103/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_206IdentityRead_103/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*
_output_shapes

:@m
Read_104/DisableCopyOnReadDisableCopyOnRead&read_104_disablecopyonread_variable_19*
_output_shapes
 �
Read_104/ReadVariableOpReadVariableOp&read_104_disablecopyonread_variable_19^Read_104/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_208IdentityRead_104/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@m
Read_105/DisableCopyOnReadDisableCopyOnRead&read_105_disablecopyonread_variable_18*
_output_shapes
 �
Read_105/ReadVariableOpReadVariableOp&read_105_disablecopyonread_variable_18^Read_105/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_210IdentityRead_105/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_211IdentityIdentity_210:output:0"/device:CPU:0*
T0*
_output_shapes

:@m
Read_106/DisableCopyOnReadDisableCopyOnRead&read_106_disablecopyonread_variable_17*
_output_shapes
 �
Read_106/ReadVariableOpReadVariableOp&read_106_disablecopyonread_variable_17^Read_106/DisableCopyOnRead*
_output_shapes
:*
dtype0^
Identity_212IdentityRead_106/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_213IdentityIdentity_212:output:0"/device:CPU:0*
T0*
_output_shapes
:m
Read_107/DisableCopyOnReadDisableCopyOnRead&read_107_disablecopyonread_variable_16*
_output_shapes
 �
Read_107/ReadVariableOpReadVariableOp&read_107_disablecopyonread_variable_16^Read_107/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0g
Identity_214IdentityRead_107/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�l
Identity_215IdentityIdentity_214:output:0"/device:CPU:0*
T0*#
_output_shapes
:@�m
Read_108/DisableCopyOnReadDisableCopyOnRead&read_108_disablecopyonread_variable_15*
_output_shapes
 �
Read_108/ReadVariableOpReadVariableOp&read_108_disablecopyonread_variable_15^Read_108/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_216IdentityRead_108/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_217IdentityIdentity_216:output:0"/device:CPU:0*
T0*
_output_shapes	
:�m
Read_109/DisableCopyOnReadDisableCopyOnRead&read_109_disablecopyonread_variable_14*
_output_shapes
 �
Read_109/ReadVariableOpReadVariableOp&read_109_disablecopyonread_variable_14^Read_109/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_218IdentityRead_109/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_219IdentityIdentity_218:output:0"/device:CPU:0*
T0*
_output_shapes	
:�m
Read_110/DisableCopyOnReadDisableCopyOnRead&read_110_disablecopyonread_variable_13*
_output_shapes
 �
Read_110/ReadVariableOpReadVariableOp&read_110_disablecopyonread_variable_13^Read_110/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_220IdentityRead_110/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_221IdentityIdentity_220:output:0"/device:CPU:0*
T0*
_output_shapes	
:�m
Read_111/DisableCopyOnReadDisableCopyOnRead&read_111_disablecopyonread_variable_12*
_output_shapes
 �
Read_111/ReadVariableOpReadVariableOp&read_111_disablecopyonread_variable_12^Read_111/DisableCopyOnRead*
_output_shapes
:*
dtype0^
Identity_222IdentityRead_111/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_223IdentityIdentity_222:output:0"/device:CPU:0*
T0*
_output_shapes
:m
Read_112/DisableCopyOnReadDisableCopyOnRead&read_112_disablecopyonread_variable_11*
_output_shapes
 �
Read_112/ReadVariableOpReadVariableOp&read_112_disablecopyonread_variable_11^Read_112/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_224IdentityRead_112/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_225IdentityIdentity_224:output:0"/device:CPU:0*
T0*
_output_shapes	
:�m
Read_113/DisableCopyOnReadDisableCopyOnRead&read_113_disablecopyonread_variable_10*
_output_shapes
 �
Read_113/ReadVariableOpReadVariableOp&read_113_disablecopyonread_variable_10^Read_113/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_226IdentityRead_113/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_227IdentityIdentity_226:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_114/DisableCopyOnReadDisableCopyOnRead%read_114_disablecopyonread_variable_9*
_output_shapes
 �
Read_114/ReadVariableOpReadVariableOp%read_114_disablecopyonread_variable_9^Read_114/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_228IdentityRead_114/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_229IdentityIdentity_228:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��l
Read_115/DisableCopyOnReadDisableCopyOnRead%read_115_disablecopyonread_variable_8*
_output_shapes
 �
Read_115/ReadVariableOpReadVariableOp%read_115_disablecopyonread_variable_8^Read_115/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_230IdentityRead_115/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_231IdentityIdentity_230:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_116/DisableCopyOnReadDisableCopyOnRead%read_116_disablecopyonread_variable_7*
_output_shapes
 �
Read_116/ReadVariableOpReadVariableOp%read_116_disablecopyonread_variable_7^Read_116/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_232IdentityRead_116/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_233IdentityIdentity_232:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��l
Read_117/DisableCopyOnReadDisableCopyOnRead%read_117_disablecopyonread_variable_6*
_output_shapes
 �
Read_117/ReadVariableOpReadVariableOp%read_117_disablecopyonread_variable_6^Read_117/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_234IdentityRead_117/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_235IdentityIdentity_234:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_118/DisableCopyOnReadDisableCopyOnRead%read_118_disablecopyonread_variable_5*
_output_shapes
 �
Read_118/ReadVariableOpReadVariableOp%read_118_disablecopyonread_variable_5^Read_118/DisableCopyOnRead*
_output_shapes
:*
dtype0^
Identity_236IdentityRead_118/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_237IdentityIdentity_236:output:0"/device:CPU:0*
T0*
_output_shapes
:l
Read_119/DisableCopyOnReadDisableCopyOnRead%read_119_disablecopyonread_variable_4*
_output_shapes
 �
Read_119/ReadVariableOpReadVariableOp%read_119_disablecopyonread_variable_4^Read_119/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_238IdentityRead_119/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_239IdentityIdentity_238:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��l
Read_120/DisableCopyOnReadDisableCopyOnRead%read_120_disablecopyonread_variable_3*
_output_shapes
 �
Read_120/ReadVariableOpReadVariableOp%read_120_disablecopyonread_variable_3^Read_120/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_240IdentityRead_120/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_241IdentityIdentity_240:output:0"/device:CPU:0*
T0*
_output_shapes	
:�l
Read_121/DisableCopyOnReadDisableCopyOnRead%read_121_disablecopyonread_variable_2*
_output_shapes
 �
Read_121/ReadVariableOpReadVariableOp%read_121_disablecopyonread_variable_2^Read_121/DisableCopyOnRead*
_output_shapes
:*
dtype0^
Identity_242IdentityRead_121/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_243IdentityIdentity_242:output:0"/device:CPU:0*
T0*
_output_shapes
:l
Read_122/DisableCopyOnReadDisableCopyOnRead%read_122_disablecopyonread_variable_1*
_output_shapes
 �
Read_122/ReadVariableOpReadVariableOp%read_122_disablecopyonread_variable_1^Read_122/DisableCopyOnRead*
_output_shapes
:	�*
dtype0c
Identity_244IdentityRead_122/ReadVariableOp:value:0*
T0*
_output_shapes
:	�h
Identity_245IdentityIdentity_244:output:0"/device:CPU:0*
T0*
_output_shapes
:	�j
Read_123/DisableCopyOnReadDisableCopyOnRead#read_123_disablecopyonread_variable*
_output_shapes
 �
Read_123/ReadVariableOpReadVariableOp#read_123_disablecopyonread_variable^Read_123/DisableCopyOnRead*
_output_shapes
:*
dtype0^
Identity_246IdentityRead_123/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_247IdentityIdentity_246:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_124/DisableCopyOnReadDisableCopyOnReadRread_124_disablecopyonread_transformer_layer_3_self_attention_layer_query_kernel_1*
_output_shapes
 �
Read_124/ReadVariableOpReadVariableOpRread_124_disablecopyonread_transformer_layer_3_self_attention_layer_query_kernel_1^Read_124/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_248IdentityRead_124/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_249IdentityIdentity_248:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_125/DisableCopyOnReadDisableCopyOnReadPread_125_disablecopyonread_transformer_layer_3_self_attention_layer_value_bias_1*
_output_shapes
 �
Read_125/ReadVariableOpReadVariableOpPread_125_disablecopyonread_transformer_layer_3_self_attention_layer_value_bias_1^Read_125/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_250IdentityRead_125/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_251IdentityIdentity_250:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_126/DisableCopyOnReadDisableCopyOnRead[read_126_disablecopyonread_transformer_layer_3_self_attention_layer_attention_output_bias_1*
_output_shapes
 �
Read_126/ReadVariableOpReadVariableOp[read_126_disablecopyonread_transformer_layer_3_self_attention_layer_attention_output_bias_1^Read_126/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_252IdentityRead_126/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_253IdentityIdentity_252:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_127/DisableCopyOnReadDisableCopyOnReadNread_127_disablecopyonread_transformer_layer_3_feedforward_output_dense_bias_1*
_output_shapes
 �
Read_127/ReadVariableOpReadVariableOpNread_127_disablecopyonread_transformer_layer_3_feedforward_output_dense_bias_1^Read_127/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_254IdentityRead_127/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_255IdentityIdentity_254:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_128/DisableCopyOnReadDisableCopyOnReadOread_128_disablecopyonread_transformer_layer_5_self_attention_layer_norm_beta_1*
_output_shapes
 �
Read_128/ReadVariableOpReadVariableOpOread_128_disablecopyonread_transformer_layer_5_self_attention_layer_norm_beta_1^Read_128/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_256IdentityRead_128/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_257IdentityIdentity_256:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_129/DisableCopyOnReadDisableCopyOnReadPread_129_disablecopyonread_transformer_layer_1_self_attention_layer_query_bias_1*
_output_shapes
 �
Read_129/ReadVariableOpReadVariableOpPread_129_disablecopyonread_transformer_layer_1_self_attention_layer_query_bias_1^Read_129/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_258IdentityRead_129/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_259IdentityIdentity_258:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_130/DisableCopyOnReadDisableCopyOnReadPread_130_disablecopyonread_transformer_layer_3_self_attention_layer_key_kernel_1*
_output_shapes
 �
Read_130/ReadVariableOpReadVariableOpPread_130_disablecopyonread_transformer_layer_3_self_attention_layer_key_kernel_1^Read_130/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_260IdentityRead_130/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_261IdentityIdentity_260:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_131/DisableCopyOnReadDisableCopyOnReadOread_131_disablecopyonread_transformer_layer_3_self_attention_layer_norm_beta_1*
_output_shapes
 �
Read_131/ReadVariableOpReadVariableOpOread_131_disablecopyonread_transformer_layer_3_self_attention_layer_norm_beta_1^Read_131/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_262IdentityRead_131/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_263IdentityIdentity_262:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_132/DisableCopyOnReadDisableCopyOnReadNread_132_disablecopyonread_transformer_layer_1_self_attention_layer_key_bias_1*
_output_shapes
 �
Read_132/ReadVariableOpReadVariableOpNread_132_disablecopyonread_transformer_layer_1_self_attention_layer_key_bias_1^Read_132/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_264IdentityRead_132/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_265IdentityIdentity_264:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_133/DisableCopyOnReadDisableCopyOnReadNread_133_disablecopyonread_transformer_layer_4_feedforward_output_dense_bias_1*
_output_shapes
 �
Read_133/ReadVariableOpReadVariableOpNread_133_disablecopyonread_transformer_layer_4_feedforward_output_dense_bias_1^Read_133/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_266IdentityRead_133/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_267IdentityIdentity_266:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_134/DisableCopyOnReadDisableCopyOnReadMread_134_disablecopyonread_transformer_layer_4_feedforward_layer_norm_gamma_1*
_output_shapes
 �
Read_134/ReadVariableOpReadVariableOpMread_134_disablecopyonread_transformer_layer_4_feedforward_layer_norm_gamma_1^Read_134/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_268IdentityRead_134/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_269IdentityIdentity_268:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_135/DisableCopyOnReadDisableCopyOnReadLread_135_disablecopyonread_transformer_layer_0_feedforward_layer_norm_beta_1*
_output_shapes
 �
Read_135/ReadVariableOpReadVariableOpLread_135_disablecopyonread_transformer_layer_0_feedforward_layer_norm_beta_1^Read_135/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_270IdentityRead_135/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_271IdentityIdentity_270:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_136/DisableCopyOnReadDisableCopyOnReadPread_136_disablecopyonread_transformer_layer_4_self_attention_layer_query_bias_1*
_output_shapes
 �
Read_136/ReadVariableOpReadVariableOpPread_136_disablecopyonread_transformer_layer_4_self_attention_layer_query_bias_1^Read_136/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_272IdentityRead_136/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_273IdentityIdentity_272:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_137/DisableCopyOnReadDisableCopyOnReadPread_137_disablecopyonread_transformer_layer_4_self_attention_layer_key_kernel_1*
_output_shapes
 �
Read_137/ReadVariableOpReadVariableOpPread_137_disablecopyonread_transformer_layer_4_self_attention_layer_key_kernel_1^Read_137/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_274IdentityRead_137/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_275IdentityIdentity_274:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_138/DisableCopyOnReadDisableCopyOnReadTread_138_disablecopyonread_token_and_position_embedding_token_embedding_embeddings_1*
_output_shapes
 �
Read_138/ReadVariableOpReadVariableOpTread_138_disablecopyonread_token_and_position_embedding_token_embedding_embeddings_1^Read_138/DisableCopyOnRead*!
_output_shapes
:���*
dtype0e
Identity_276IdentityRead_138/ReadVariableOp:value:0*
T0*!
_output_shapes
:���j
Identity_277IdentityIdentity_276:output:0"/device:CPU:0*
T0*!
_output_shapes
:����
Read_139/DisableCopyOnReadDisableCopyOnReadNread_139_disablecopyonread_transformer_layer_5_feedforward_output_dense_bias_1*
_output_shapes
 �
Read_139/ReadVariableOpReadVariableOpNread_139_disablecopyonread_transformer_layer_5_feedforward_output_dense_bias_1^Read_139/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_278IdentityRead_139/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_279IdentityIdentity_278:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_140/DisableCopyOnReadDisableCopyOnReadPread_140_disablecopyonread_transformer_layer_3_self_attention_layer_query_bias_1*
_output_shapes
 �
Read_140/ReadVariableOpReadVariableOpPread_140_disablecopyonread_transformer_layer_3_self_attention_layer_query_bias_1^Read_140/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_280IdentityRead_140/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_281IdentityIdentity_280:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_141/DisableCopyOnReadDisableCopyOnReadNread_141_disablecopyonread_transformer_layer_3_self_attention_layer_key_bias_1*
_output_shapes
 �
Read_141/ReadVariableOpReadVariableOpNread_141_disablecopyonread_transformer_layer_3_self_attention_layer_key_bias_1^Read_141/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_282IdentityRead_141/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_283IdentityIdentity_282:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_142/DisableCopyOnReadDisableCopyOnReadPread_142_disablecopyonread_transformer_layer_0_self_attention_layer_norm_gamma_1*
_output_shapes
 �
Read_142/ReadVariableOpReadVariableOpPread_142_disablecopyonread_transformer_layer_0_self_attention_layer_norm_gamma_1^Read_142/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_284IdentityRead_142/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_285IdentityIdentity_284:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_143/DisableCopyOnReadDisableCopyOnReadLread_143_disablecopyonread_transformer_layer_4_feedforward_layer_norm_beta_1*
_output_shapes
 �
Read_143/ReadVariableOpReadVariableOpLread_143_disablecopyonread_transformer_layer_4_feedforward_layer_norm_beta_1^Read_143/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_286IdentityRead_143/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_287IdentityIdentity_286:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_144/DisableCopyOnReadDisableCopyOnRead7read_144_disablecopyonread_embeddings_layer_norm_beta_1*
_output_shapes
 �
Read_144/ReadVariableOpReadVariableOp7read_144_disablecopyonread_embeddings_layer_norm_beta_1^Read_144/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_288IdentityRead_144/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_289IdentityIdentity_288:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_145/DisableCopyOnReadDisableCopyOnReadRread_145_disablecopyonread_transformer_layer_2_self_attention_layer_query_kernel_1*
_output_shapes
 �
Read_145/ReadVariableOpReadVariableOpRread_145_disablecopyonread_transformer_layer_2_self_attention_layer_query_kernel_1^Read_145/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_290IdentityRead_145/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_291IdentityIdentity_290:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_146/DisableCopyOnReadDisableCopyOnReadMread_146_disablecopyonread_transformer_layer_2_feedforward_layer_norm_gamma_1*
_output_shapes
 �
Read_146/ReadVariableOpReadVariableOpMread_146_disablecopyonread_transformer_layer_2_feedforward_layer_norm_gamma_1^Read_146/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_292IdentityRead_146/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_293IdentityIdentity_292:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_147/DisableCopyOnReadDisableCopyOnReadTread_147_disablecopyonread_transformer_layer_4_feedforward_intermediate_dense_bias_1*
_output_shapes
 �
Read_147/ReadVariableOpReadVariableOpTread_147_disablecopyonread_transformer_layer_4_feedforward_intermediate_dense_bias_1^Read_147/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_294IdentityRead_147/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_295IdentityIdentity_294:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_148/DisableCopyOnReadDisableCopyOnReadNread_148_disablecopyonread_transformer_layer_4_self_attention_layer_key_bias_1*
_output_shapes
 �
Read_148/ReadVariableOpReadVariableOpNread_148_disablecopyonread_transformer_layer_4_self_attention_layer_key_bias_1^Read_148/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_296IdentityRead_148/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_297IdentityIdentity_296:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_149/DisableCopyOnReadDisableCopyOnReadRread_149_disablecopyonread_transformer_layer_4_self_attention_layer_value_kernel_1*
_output_shapes
 �
Read_149/ReadVariableOpReadVariableOpRread_149_disablecopyonread_transformer_layer_4_self_attention_layer_value_kernel_1^Read_149/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_298IdentityRead_149/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_299IdentityIdentity_298:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_150/DisableCopyOnReadDisableCopyOnReadPread_150_disablecopyonread_transformer_layer_1_feedforward_output_dense_kernel_1*
_output_shapes
 �
Read_150/ReadVariableOpReadVariableOpPread_150_disablecopyonread_transformer_layer_1_feedforward_output_dense_kernel_1^Read_150/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_300IdentityRead_150/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_301IdentityIdentity_300:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_151/DisableCopyOnReadDisableCopyOnRead]read_151_disablecopyonread_transformer_layer_0_self_attention_layer_attention_output_kernel_1*
_output_shapes
 �
Read_151/ReadVariableOpReadVariableOp]read_151_disablecopyonread_transformer_layer_0_self_attention_layer_attention_output_kernel_1^Read_151/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0g
Identity_302IdentityRead_151/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�l
Identity_303IdentityIdentity_302:output:0"/device:CPU:0*
T0*#
_output_shapes
:@��
Read_152/DisableCopyOnReadDisableCopyOnReadMread_152_disablecopyonread_transformer_layer_1_feedforward_layer_norm_gamma_1*
_output_shapes
 �
Read_152/ReadVariableOpReadVariableOpMread_152_disablecopyonread_transformer_layer_1_feedforward_layer_norm_gamma_1^Read_152/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_304IdentityRead_152/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_305IdentityIdentity_304:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_153/DisableCopyOnReadDisableCopyOnReadTread_153_disablecopyonread_transformer_layer_1_feedforward_intermediate_dense_bias_1*
_output_shapes
 �
Read_153/ReadVariableOpReadVariableOpTread_153_disablecopyonread_transformer_layer_1_feedforward_intermediate_dense_bias_1^Read_153/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_306IdentityRead_153/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_307IdentityIdentity_306:output:0"/device:CPU:0*
T0*
_output_shapes	
:�u
Read_154/DisableCopyOnReadDisableCopyOnRead.read_154_disablecopyonread_pooled_dense_bias_1*
_output_shapes
 �
Read_154/ReadVariableOpReadVariableOp.read_154_disablecopyonread_pooled_dense_bias_1^Read_154/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_308IdentityRead_154/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_309IdentityIdentity_308:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_155/DisableCopyOnReadDisableCopyOnReadPread_155_disablecopyonread_transformer_layer_2_self_attention_layer_query_bias_1*
_output_shapes
 �
Read_155/ReadVariableOpReadVariableOpPread_155_disablecopyonread_transformer_layer_2_self_attention_layer_query_bias_1^Read_155/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_310IdentityRead_155/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_311IdentityIdentity_310:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_156/DisableCopyOnReadDisableCopyOnReadLread_156_disablecopyonread_transformer_layer_2_feedforward_layer_norm_beta_1*
_output_shapes
 �
Read_156/ReadVariableOpReadVariableOpLread_156_disablecopyonread_transformer_layer_2_feedforward_layer_norm_beta_1^Read_156/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_312IdentityRead_156/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_313IdentityIdentity_312:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_157/DisableCopyOnReadDisableCopyOnReadVread_157_disablecopyonread_transformer_layer_4_feedforward_intermediate_dense_kernel_1*
_output_shapes
 �
Read_157/ReadVariableOpReadVariableOpVread_157_disablecopyonread_transformer_layer_4_feedforward_intermediate_dense_kernel_1^Read_157/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_314IdentityRead_157/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_315IdentityIdentity_314:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_158/DisableCopyOnReadDisableCopyOnRead]read_158_disablecopyonread_transformer_layer_4_self_attention_layer_attention_output_kernel_1*
_output_shapes
 �
Read_158/ReadVariableOpReadVariableOp]read_158_disablecopyonread_transformer_layer_4_self_attention_layer_attention_output_kernel_1^Read_158/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0g
Identity_316IdentityRead_158/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�l
Identity_317IdentityIdentity_316:output:0"/device:CPU:0*
T0*#
_output_shapes
:@��
Read_159/DisableCopyOnReadDisableCopyOnReadWread_159_disablecopyonread_token_and_position_embedding_position_embedding_embeddings_1*
_output_shapes
 �
Read_159/ReadVariableOpReadVariableOpWread_159_disablecopyonread_token_and_position_embedding_position_embedding_embeddings_1^Read_159/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_318IdentityRead_159/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_319IdentityIdentity_318:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_160/DisableCopyOnReadDisableCopyOnReadPread_160_disablecopyonread_transformer_layer_2_self_attention_layer_key_kernel_1*
_output_shapes
 �
Read_160/ReadVariableOpReadVariableOpPread_160_disablecopyonread_transformer_layer_2_self_attention_layer_key_kernel_1^Read_160/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_320IdentityRead_160/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_321IdentityIdentity_320:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_161/DisableCopyOnReadDisableCopyOnReadTread_161_disablecopyonread_transformer_layer_2_feedforward_intermediate_dense_bias_1*
_output_shapes
 �
Read_161/ReadVariableOpReadVariableOpTread_161_disablecopyonread_transformer_layer_2_feedforward_intermediate_dense_bias_1^Read_161/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_322IdentityRead_161/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_323IdentityIdentity_322:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_162/DisableCopyOnReadDisableCopyOnReadPread_162_disablecopyonread_transformer_layer_4_self_attention_layer_value_bias_1*
_output_shapes
 �
Read_162/ReadVariableOpReadVariableOpPread_162_disablecopyonread_transformer_layer_4_self_attention_layer_value_bias_1^Read_162/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_324IdentityRead_162/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_325IdentityIdentity_324:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_163/DisableCopyOnReadDisableCopyOnReadMread_163_disablecopyonread_transformer_layer_5_feedforward_layer_norm_gamma_1*
_output_shapes
 �
Read_163/ReadVariableOpReadVariableOpMread_163_disablecopyonread_transformer_layer_5_feedforward_layer_norm_gamma_1^Read_163/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_326IdentityRead_163/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_327IdentityIdentity_326:output:0"/device:CPU:0*
T0*
_output_shapes	
:�q
Read_164/DisableCopyOnReadDisableCopyOnRead*read_164_disablecopyonread_logits_kernel_1*
_output_shapes
 �
Read_164/ReadVariableOpReadVariableOp*read_164_disablecopyonread_logits_kernel_1^Read_164/DisableCopyOnRead*
_output_shapes
:	�*
dtype0c
Identity_328IdentityRead_164/ReadVariableOp:value:0*
T0*
_output_shapes
:	�h
Identity_329IdentityIdentity_328:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_165/DisableCopyOnReadDisableCopyOnRead8read_165_disablecopyonread_embeddings_layer_norm_gamma_1*
_output_shapes
 �
Read_165/ReadVariableOpReadVariableOp8read_165_disablecopyonread_embeddings_layer_norm_gamma_1^Read_165/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_330IdentityRead_165/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_331IdentityIdentity_330:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_166/DisableCopyOnReadDisableCopyOnReadRread_166_disablecopyonread_transformer_layer_2_self_attention_layer_value_kernel_1*
_output_shapes
 �
Read_166/ReadVariableOpReadVariableOpRread_166_disablecopyonread_transformer_layer_2_self_attention_layer_value_kernel_1^Read_166/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_332IdentityRead_166/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_333IdentityIdentity_332:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_167/DisableCopyOnReadDisableCopyOnReadMread_167_disablecopyonread_transformer_layer_3_feedforward_layer_norm_gamma_1*
_output_shapes
 �
Read_167/ReadVariableOpReadVariableOpMread_167_disablecopyonread_transformer_layer_3_feedforward_layer_norm_gamma_1^Read_167/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_334IdentityRead_167/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_335IdentityIdentity_334:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_168/DisableCopyOnReadDisableCopyOnReadPread_168_disablecopyonread_transformer_layer_4_self_attention_layer_norm_gamma_1*
_output_shapes
 �
Read_168/ReadVariableOpReadVariableOpPread_168_disablecopyonread_transformer_layer_4_self_attention_layer_norm_gamma_1^Read_168/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_336IdentityRead_168/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_337IdentityIdentity_336:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_169/DisableCopyOnReadDisableCopyOnReadRread_169_disablecopyonread_transformer_layer_5_self_attention_layer_query_kernel_1*
_output_shapes
 �
Read_169/ReadVariableOpReadVariableOpRread_169_disablecopyonread_transformer_layer_5_self_attention_layer_query_kernel_1^Read_169/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_338IdentityRead_169/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_339IdentityIdentity_338:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_170/DisableCopyOnReadDisableCopyOnReadRread_170_disablecopyonread_transformer_layer_0_self_attention_layer_query_kernel_1*
_output_shapes
 �
Read_170/ReadVariableOpReadVariableOpRread_170_disablecopyonread_transformer_layer_0_self_attention_layer_query_kernel_1^Read_170/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_340IdentityRead_170/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_341IdentityIdentity_340:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_171/DisableCopyOnReadDisableCopyOnRead[read_171_disablecopyonread_transformer_layer_0_self_attention_layer_attention_output_bias_1*
_output_shapes
 �
Read_171/ReadVariableOpReadVariableOp[read_171_disablecopyonread_transformer_layer_0_self_attention_layer_attention_output_bias_1^Read_171/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_342IdentityRead_171/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_343IdentityIdentity_342:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_172/DisableCopyOnReadDisableCopyOnReadLread_172_disablecopyonread_transformer_layer_1_feedforward_layer_norm_beta_1*
_output_shapes
 �
Read_172/ReadVariableOpReadVariableOpLread_172_disablecopyonread_transformer_layer_1_feedforward_layer_norm_beta_1^Read_172/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_344IdentityRead_172/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_345IdentityIdentity_344:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_173/DisableCopyOnReadDisableCopyOnReadPread_173_disablecopyonread_transformer_layer_5_self_attention_layer_key_kernel_1*
_output_shapes
 �
Read_173/ReadVariableOpReadVariableOpPread_173_disablecopyonread_transformer_layer_5_self_attention_layer_key_kernel_1^Read_173/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_346IdentityRead_173/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_347IdentityIdentity_346:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_174/DisableCopyOnReadDisableCopyOnReadPread_174_disablecopyonread_transformer_layer_0_self_attention_layer_key_kernel_1*
_output_shapes
 �
Read_174/ReadVariableOpReadVariableOpPread_174_disablecopyonread_transformer_layer_0_self_attention_layer_key_kernel_1^Read_174/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_348IdentityRead_174/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_349IdentityIdentity_348:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_175/DisableCopyOnReadDisableCopyOnReadOread_175_disablecopyonread_transformer_layer_0_self_attention_layer_norm_beta_1*
_output_shapes
 �
Read_175/ReadVariableOpReadVariableOpOread_175_disablecopyonread_transformer_layer_0_self_attention_layer_norm_beta_1^Read_175/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_350IdentityRead_175/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_351IdentityIdentity_350:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_176/DisableCopyOnReadDisableCopyOnReadRread_176_disablecopyonread_transformer_layer_5_self_attention_layer_value_kernel_1*
_output_shapes
 �
Read_176/ReadVariableOpReadVariableOpRread_176_disablecopyonread_transformer_layer_5_self_attention_layer_value_kernel_1^Read_176/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_352IdentityRead_176/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_353IdentityIdentity_352:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_177/DisableCopyOnReadDisableCopyOnReadRread_177_disablecopyonread_transformer_layer_1_self_attention_layer_value_kernel_1*
_output_shapes
 �
Read_177/ReadVariableOpReadVariableOpRread_177_disablecopyonread_transformer_layer_1_self_attention_layer_value_kernel_1^Read_177/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_354IdentityRead_177/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_355IdentityIdentity_354:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_178/DisableCopyOnReadDisableCopyOnReadNread_178_disablecopyonread_transformer_layer_2_self_attention_layer_key_bias_1*
_output_shapes
 �
Read_178/ReadVariableOpReadVariableOpNread_178_disablecopyonread_transformer_layer_2_self_attention_layer_key_bias_1^Read_178/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_356IdentityRead_178/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_357IdentityIdentity_356:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_179/DisableCopyOnReadDisableCopyOnReadVread_179_disablecopyonread_transformer_layer_2_feedforward_intermediate_dense_kernel_1*
_output_shapes
 �
Read_179/ReadVariableOpReadVariableOpVread_179_disablecopyonread_transformer_layer_2_feedforward_intermediate_dense_kernel_1^Read_179/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_358IdentityRead_179/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_359IdentityIdentity_358:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_180/DisableCopyOnReadDisableCopyOnRead[read_180_disablecopyonread_transformer_layer_4_self_attention_layer_attention_output_bias_1*
_output_shapes
 �
Read_180/ReadVariableOpReadVariableOp[read_180_disablecopyonread_transformer_layer_4_self_attention_layer_attention_output_bias_1^Read_180/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_360IdentityRead_180/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_361IdentityIdentity_360:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_181/DisableCopyOnReadDisableCopyOnReadLread_181_disablecopyonread_transformer_layer_5_feedforward_layer_norm_beta_1*
_output_shapes
 �
Read_181/ReadVariableOpReadVariableOpLread_181_disablecopyonread_transformer_layer_5_feedforward_layer_norm_beta_1^Read_181/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_362IdentityRead_181/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_363IdentityIdentity_362:output:0"/device:CPU:0*
T0*
_output_shapes	
:�w
Read_182/DisableCopyOnReadDisableCopyOnRead0read_182_disablecopyonread_pooled_dense_kernel_1*
_output_shapes
 �
Read_182/ReadVariableOpReadVariableOp0read_182_disablecopyonread_pooled_dense_kernel_1^Read_182/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_364IdentityRead_182/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_365IdentityIdentity_364:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��o
Read_183/DisableCopyOnReadDisableCopyOnRead(read_183_disablecopyonread_logits_bias_1*
_output_shapes
 �
Read_183/ReadVariableOpReadVariableOp(read_183_disablecopyonread_logits_bias_1^Read_183/DisableCopyOnRead*
_output_shapes
:*
dtype0^
Identity_366IdentityRead_183/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_367IdentityIdentity_366:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_184/DisableCopyOnReadDisableCopyOnReadNread_184_disablecopyonread_transformer_layer_0_feedforward_output_dense_bias_1*
_output_shapes
 �
Read_184/ReadVariableOpReadVariableOpNread_184_disablecopyonread_transformer_layer_0_feedforward_output_dense_bias_1^Read_184/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_368IdentityRead_184/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_369IdentityIdentity_368:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_185/DisableCopyOnReadDisableCopyOnReadPread_185_disablecopyonread_transformer_layer_2_self_attention_layer_value_bias_1*
_output_shapes
 �
Read_185/ReadVariableOpReadVariableOpPread_185_disablecopyonread_transformer_layer_2_self_attention_layer_value_bias_1^Read_185/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_370IdentityRead_185/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_371IdentityIdentity_370:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_186/DisableCopyOnReadDisableCopyOnRead]read_186_disablecopyonread_transformer_layer_2_self_attention_layer_attention_output_kernel_1*
_output_shapes
 �
Read_186/ReadVariableOpReadVariableOp]read_186_disablecopyonread_transformer_layer_2_self_attention_layer_attention_output_kernel_1^Read_186/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0g
Identity_372IdentityRead_186/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�l
Identity_373IdentityIdentity_372:output:0"/device:CPU:0*
T0*#
_output_shapes
:@��
Read_187/DisableCopyOnReadDisableCopyOnReadLread_187_disablecopyonread_transformer_layer_3_feedforward_layer_norm_beta_1*
_output_shapes
 �
Read_187/ReadVariableOpReadVariableOpLread_187_disablecopyonread_transformer_layer_3_feedforward_layer_norm_beta_1^Read_187/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_374IdentityRead_187/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_375IdentityIdentity_374:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_188/DisableCopyOnReadDisableCopyOnReadOread_188_disablecopyonread_transformer_layer_4_self_attention_layer_norm_beta_1*
_output_shapes
 �
Read_188/ReadVariableOpReadVariableOpOread_188_disablecopyonread_transformer_layer_4_self_attention_layer_norm_beta_1^Read_188/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_376IdentityRead_188/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_377IdentityIdentity_376:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_189/DisableCopyOnReadDisableCopyOnReadPread_189_disablecopyonread_transformer_layer_5_self_attention_layer_query_bias_1*
_output_shapes
 �
Read_189/ReadVariableOpReadVariableOpPread_189_disablecopyonread_transformer_layer_5_self_attention_layer_query_bias_1^Read_189/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_378IdentityRead_189/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_379IdentityIdentity_378:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_190/DisableCopyOnReadDisableCopyOnReadVread_190_disablecopyonread_transformer_layer_5_feedforward_intermediate_dense_kernel_1*
_output_shapes
 �
Read_190/ReadVariableOpReadVariableOpVread_190_disablecopyonread_transformer_layer_5_feedforward_intermediate_dense_kernel_1^Read_190/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_380IdentityRead_190/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_381IdentityIdentity_380:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_191/DisableCopyOnReadDisableCopyOnReadPread_191_disablecopyonread_transformer_layer_0_self_attention_layer_query_bias_1*
_output_shapes
 �
Read_191/ReadVariableOpReadVariableOpPread_191_disablecopyonread_transformer_layer_0_self_attention_layer_query_bias_1^Read_191/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_382IdentityRead_191/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_383IdentityIdentity_382:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_192/DisableCopyOnReadDisableCopyOnReadPread_192_disablecopyonread_transformer_layer_2_self_attention_layer_norm_gamma_1*
_output_shapes
 �
Read_192/ReadVariableOpReadVariableOpPread_192_disablecopyonread_transformer_layer_2_self_attention_layer_norm_gamma_1^Read_192/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_384IdentityRead_192/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_385IdentityIdentity_384:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_193/DisableCopyOnReadDisableCopyOnReadVread_193_disablecopyonread_transformer_layer_3_feedforward_intermediate_dense_kernel_1*
_output_shapes
 �
Read_193/ReadVariableOpReadVariableOpVread_193_disablecopyonread_transformer_layer_3_feedforward_intermediate_dense_kernel_1^Read_193/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_386IdentityRead_193/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_387IdentityIdentity_386:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_194/DisableCopyOnReadDisableCopyOnReadNread_194_disablecopyonread_transformer_layer_0_self_attention_layer_key_bias_1*
_output_shapes
 �
Read_194/ReadVariableOpReadVariableOpNread_194_disablecopyonread_transformer_layer_0_self_attention_layer_key_bias_1^Read_194/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_388IdentityRead_194/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_389IdentityIdentity_388:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_195/DisableCopyOnReadDisableCopyOnReadVread_195_disablecopyonread_transformer_layer_1_feedforward_intermediate_dense_kernel_1*
_output_shapes
 �
Read_195/ReadVariableOpReadVariableOpVread_195_disablecopyonread_transformer_layer_1_feedforward_intermediate_dense_kernel_1^Read_195/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_390IdentityRead_195/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_391IdentityIdentity_390:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_196/DisableCopyOnReadDisableCopyOnReadPread_196_disablecopyonread_transformer_layer_5_self_attention_layer_value_bias_1*
_output_shapes
 �
Read_196/ReadVariableOpReadVariableOpPread_196_disablecopyonread_transformer_layer_5_self_attention_layer_value_bias_1^Read_196/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_392IdentityRead_196/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_393IdentityIdentity_392:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_197/DisableCopyOnReadDisableCopyOnRead]read_197_disablecopyonread_transformer_layer_5_self_attention_layer_attention_output_kernel_1*
_output_shapes
 �
Read_197/ReadVariableOpReadVariableOp]read_197_disablecopyonread_transformer_layer_5_self_attention_layer_attention_output_kernel_1^Read_197/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0g
Identity_394IdentityRead_197/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�l
Identity_395IdentityIdentity_394:output:0"/device:CPU:0*
T0*#
_output_shapes
:@��
Read_198/DisableCopyOnReadDisableCopyOnReadRread_198_disablecopyonread_transformer_layer_0_self_attention_layer_value_kernel_1*
_output_shapes
 �
Read_198/ReadVariableOpReadVariableOpRread_198_disablecopyonread_transformer_layer_0_self_attention_layer_value_kernel_1^Read_198/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_396IdentityRead_198/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_397IdentityIdentity_396:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_199/DisableCopyOnReadDisableCopyOnReadPread_199_disablecopyonread_transformer_layer_1_self_attention_layer_value_bias_1*
_output_shapes
 �
Read_199/ReadVariableOpReadVariableOpPread_199_disablecopyonread_transformer_layer_1_self_attention_layer_value_bias_1^Read_199/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_398IdentityRead_199/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_399IdentityIdentity_398:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_200/DisableCopyOnReadDisableCopyOnRead]read_200_disablecopyonread_transformer_layer_1_self_attention_layer_attention_output_kernel_1*
_output_shapes
 �
Read_200/ReadVariableOpReadVariableOp]read_200_disablecopyonread_transformer_layer_1_self_attention_layer_attention_output_kernel_1^Read_200/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0g
Identity_400IdentityRead_200/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�l
Identity_401IdentityIdentity_400:output:0"/device:CPU:0*
T0*#
_output_shapes
:@��
Read_201/DisableCopyOnReadDisableCopyOnReadVread_201_disablecopyonread_transformer_layer_0_feedforward_intermediate_dense_kernel_1*
_output_shapes
 �
Read_201/ReadVariableOpReadVariableOpVread_201_disablecopyonread_transformer_layer_0_feedforward_intermediate_dense_kernel_1^Read_201/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_402IdentityRead_201/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_403IdentityIdentity_402:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_202/DisableCopyOnReadDisableCopyOnReadPread_202_disablecopyonread_transformer_layer_1_self_attention_layer_norm_gamma_1*
_output_shapes
 �
Read_202/ReadVariableOpReadVariableOpPread_202_disablecopyonread_transformer_layer_1_self_attention_layer_norm_gamma_1^Read_202/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_404IdentityRead_202/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_405IdentityIdentity_404:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_203/DisableCopyOnReadDisableCopyOnRead[read_203_disablecopyonread_transformer_layer_2_self_attention_layer_attention_output_bias_1*
_output_shapes
 �
Read_203/ReadVariableOpReadVariableOp[read_203_disablecopyonread_transformer_layer_2_self_attention_layer_attention_output_bias_1^Read_203/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_406IdentityRead_203/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_407IdentityIdentity_406:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_204/DisableCopyOnReadDisableCopyOnReadPread_204_disablecopyonread_transformer_layer_2_feedforward_output_dense_kernel_1*
_output_shapes
 �
Read_204/ReadVariableOpReadVariableOpPread_204_disablecopyonread_transformer_layer_2_feedforward_output_dense_kernel_1^Read_204/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_408IdentityRead_204/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_409IdentityIdentity_408:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_205/DisableCopyOnReadDisableCopyOnReadTread_205_disablecopyonread_transformer_layer_5_feedforward_intermediate_dense_bias_1*
_output_shapes
 �
Read_205/ReadVariableOpReadVariableOpTread_205_disablecopyonread_transformer_layer_5_feedforward_intermediate_dense_bias_1^Read_205/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_410IdentityRead_205/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_411IdentityIdentity_410:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_206/DisableCopyOnReadDisableCopyOnReadOread_206_disablecopyonread_transformer_layer_2_self_attention_layer_norm_beta_1*
_output_shapes
 �
Read_206/ReadVariableOpReadVariableOpOread_206_disablecopyonread_transformer_layer_2_self_attention_layer_norm_beta_1^Read_206/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_412IdentityRead_206/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_413IdentityIdentity_412:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_207/DisableCopyOnReadDisableCopyOnReadTread_207_disablecopyonread_transformer_layer_3_feedforward_intermediate_dense_bias_1*
_output_shapes
 �
Read_207/ReadVariableOpReadVariableOpTread_207_disablecopyonread_transformer_layer_3_feedforward_intermediate_dense_bias_1^Read_207/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_414IdentityRead_207/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_415IdentityIdentity_414:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_208/DisableCopyOnReadDisableCopyOnReadNread_208_disablecopyonread_transformer_layer_5_self_attention_layer_key_bias_1*
_output_shapes
 �
Read_208/ReadVariableOpReadVariableOpNread_208_disablecopyonread_transformer_layer_5_self_attention_layer_key_bias_1^Read_208/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_416IdentityRead_208/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_417IdentityIdentity_416:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_209/DisableCopyOnReadDisableCopyOnReadNread_209_disablecopyonread_transformer_layer_1_feedforward_output_dense_bias_1*
_output_shapes
 �
Read_209/ReadVariableOpReadVariableOpNread_209_disablecopyonread_transformer_layer_1_feedforward_output_dense_bias_1^Read_209/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_418IdentityRead_209/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_419IdentityIdentity_418:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_210/DisableCopyOnReadDisableCopyOnReadRread_210_disablecopyonread_transformer_layer_3_self_attention_layer_value_kernel_1*
_output_shapes
 �
Read_210/ReadVariableOpReadVariableOpRread_210_disablecopyonread_transformer_layer_3_self_attention_layer_value_kernel_1^Read_210/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_420IdentityRead_210/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_421IdentityIdentity_420:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_211/DisableCopyOnReadDisableCopyOnRead]read_211_disablecopyonread_transformer_layer_3_self_attention_layer_attention_output_kernel_1*
_output_shapes
 �
Read_211/ReadVariableOpReadVariableOp]read_211_disablecopyonread_transformer_layer_3_self_attention_layer_attention_output_kernel_1^Read_211/DisableCopyOnRead*#
_output_shapes
:@�*
dtype0g
Identity_422IdentityRead_211/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�l
Identity_423IdentityIdentity_422:output:0"/device:CPU:0*
T0*#
_output_shapes
:@��
Read_212/DisableCopyOnReadDisableCopyOnReadPread_212_disablecopyonread_transformer_layer_3_feedforward_output_dense_kernel_1*
_output_shapes
 �
Read_212/ReadVariableOpReadVariableOpPread_212_disablecopyonread_transformer_layer_3_feedforward_output_dense_kernel_1^Read_212/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_424IdentityRead_212/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_425IdentityIdentity_424:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_213/DisableCopyOnReadDisableCopyOnRead[read_213_disablecopyonread_transformer_layer_5_self_attention_layer_attention_output_bias_1*
_output_shapes
 �
Read_213/ReadVariableOpReadVariableOp[read_213_disablecopyonread_transformer_layer_5_self_attention_layer_attention_output_bias_1^Read_213/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_426IdentityRead_213/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_427IdentityIdentity_426:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_214/DisableCopyOnReadDisableCopyOnReadPread_214_disablecopyonread_transformer_layer_5_self_attention_layer_norm_gamma_1*
_output_shapes
 �
Read_214/ReadVariableOpReadVariableOpPread_214_disablecopyonread_transformer_layer_5_self_attention_layer_norm_gamma_1^Read_214/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_428IdentityRead_214/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_429IdentityIdentity_428:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_215/DisableCopyOnReadDisableCopyOnReadPread_215_disablecopyonread_transformer_layer_0_self_attention_layer_value_bias_1*
_output_shapes
 �
Read_215/ReadVariableOpReadVariableOpPread_215_disablecopyonread_transformer_layer_0_self_attention_layer_value_bias_1^Read_215/DisableCopyOnRead*
_output_shapes

:@*
dtype0b
Identity_430IdentityRead_215/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
Identity_431IdentityIdentity_430:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_216/DisableCopyOnReadDisableCopyOnReadRread_216_disablecopyonread_transformer_layer_1_self_attention_layer_query_kernel_1*
_output_shapes
 �
Read_216/ReadVariableOpReadVariableOpRread_216_disablecopyonread_transformer_layer_1_self_attention_layer_query_kernel_1^Read_216/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_432IdentityRead_216/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_433IdentityIdentity_432:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_217/DisableCopyOnReadDisableCopyOnRead[read_217_disablecopyonread_transformer_layer_1_self_attention_layer_attention_output_bias_1*
_output_shapes
 �
Read_217/ReadVariableOpReadVariableOp[read_217_disablecopyonread_transformer_layer_1_self_attention_layer_attention_output_bias_1^Read_217/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_434IdentityRead_217/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_435IdentityIdentity_434:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_218/DisableCopyOnReadDisableCopyOnReadPread_218_disablecopyonread_transformer_layer_3_self_attention_layer_norm_gamma_1*
_output_shapes
 �
Read_218/ReadVariableOpReadVariableOpPread_218_disablecopyonread_transformer_layer_3_self_attention_layer_norm_gamma_1^Read_218/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_436IdentityRead_218/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_437IdentityIdentity_436:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_219/DisableCopyOnReadDisableCopyOnReadTread_219_disablecopyonread_transformer_layer_0_feedforward_intermediate_dense_bias_1*
_output_shapes
 �
Read_219/ReadVariableOpReadVariableOpTread_219_disablecopyonread_transformer_layer_0_feedforward_intermediate_dense_bias_1^Read_219/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_438IdentityRead_219/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_439IdentityIdentity_438:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_220/DisableCopyOnReadDisableCopyOnReadPread_220_disablecopyonread_transformer_layer_1_self_attention_layer_key_kernel_1*
_output_shapes
 �
Read_220/ReadVariableOpReadVariableOpPread_220_disablecopyonread_transformer_layer_1_self_attention_layer_key_kernel_1^Read_220/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_440IdentityRead_220/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_441IdentityIdentity_440:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_221/DisableCopyOnReadDisableCopyOnReadOread_221_disablecopyonread_transformer_layer_1_self_attention_layer_norm_beta_1*
_output_shapes
 �
Read_221/ReadVariableOpReadVariableOpOread_221_disablecopyonread_transformer_layer_1_self_attention_layer_norm_beta_1^Read_221/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_442IdentityRead_221/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_443IdentityIdentity_442:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_222/DisableCopyOnReadDisableCopyOnReadPread_222_disablecopyonread_transformer_layer_4_feedforward_output_dense_kernel_1*
_output_shapes
 �
Read_222/ReadVariableOpReadVariableOpPread_222_disablecopyonread_transformer_layer_4_feedforward_output_dense_kernel_1^Read_222/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_444IdentityRead_222/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_445IdentityIdentity_444:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_223/DisableCopyOnReadDisableCopyOnReadRread_223_disablecopyonread_transformer_layer_4_self_attention_layer_query_kernel_1*
_output_shapes
 �
Read_223/ReadVariableOpReadVariableOpRread_223_disablecopyonread_transformer_layer_4_self_attention_layer_query_kernel_1^Read_223/DisableCopyOnRead*#
_output_shapes
:�@*
dtype0g
Identity_446IdentityRead_223/ReadVariableOp:value:0*
T0*#
_output_shapes
:�@l
Identity_447IdentityIdentity_446:output:0"/device:CPU:0*
T0*#
_output_shapes
:�@�
Read_224/DisableCopyOnReadDisableCopyOnReadPread_224_disablecopyonread_transformer_layer_0_feedforward_output_dense_kernel_1*
_output_shapes
 �
Read_224/ReadVariableOpReadVariableOpPread_224_disablecopyonread_transformer_layer_0_feedforward_output_dense_kernel_1^Read_224/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_448IdentityRead_224/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_449IdentityIdentity_448:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_225/DisableCopyOnReadDisableCopyOnReadMread_225_disablecopyonread_transformer_layer_0_feedforward_layer_norm_gamma_1*
_output_shapes
 �
Read_225/ReadVariableOpReadVariableOpMread_225_disablecopyonread_transformer_layer_0_feedforward_layer_norm_gamma_1^Read_225/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_450IdentityRead_225/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_451IdentityIdentity_450:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_226/DisableCopyOnReadDisableCopyOnReadNread_226_disablecopyonread_transformer_layer_2_feedforward_output_dense_bias_1*
_output_shapes
 �
Read_226/ReadVariableOpReadVariableOpNread_226_disablecopyonread_transformer_layer_2_feedforward_output_dense_bias_1^Read_226/DisableCopyOnRead*
_output_shapes	
:�*
dtype0_
Identity_452IdentityRead_226/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_453IdentityIdentity_452:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_227/DisableCopyOnReadDisableCopyOnReadPread_227_disablecopyonread_transformer_layer_5_feedforward_output_dense_kernel_1*
_output_shapes
 �
Read_227/ReadVariableOpReadVariableOpPread_227_disablecopyonread_transformer_layer_5_feedforward_output_dense_kernel_1^Read_227/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0d
Identity_454IdentityRead_227/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_455IdentityIdentity_454:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �N
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�M
value�MB�M�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB'variables/56/.ATTRIBUTES/VARIABLE_VALUEB'variables/57/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB'variables/60/.ATTRIBUTES/VARIABLE_VALUEB'variables/61/.ATTRIBUTES/VARIABLE_VALUEB'variables/62/.ATTRIBUTES/VARIABLE_VALUEB'variables/63/.ATTRIBUTES/VARIABLE_VALUEB'variables/64/.ATTRIBUTES/VARIABLE_VALUEB'variables/65/.ATTRIBUTES/VARIABLE_VALUEB'variables/66/.ATTRIBUTES/VARIABLE_VALUEB'variables/67/.ATTRIBUTES/VARIABLE_VALUEB'variables/68/.ATTRIBUTES/VARIABLE_VALUEB'variables/69/.ATTRIBUTES/VARIABLE_VALUEB'variables/70/.ATTRIBUTES/VARIABLE_VALUEB'variables/71/.ATTRIBUTES/VARIABLE_VALUEB'variables/72/.ATTRIBUTES/VARIABLE_VALUEB'variables/73/.ATTRIBUTES/VARIABLE_VALUEB'variables/74/.ATTRIBUTES/VARIABLE_VALUEB'variables/75/.ATTRIBUTES/VARIABLE_VALUEB'variables/76/.ATTRIBUTES/VARIABLE_VALUEB'variables/77/.ATTRIBUTES/VARIABLE_VALUEB'variables/78/.ATTRIBUTES/VARIABLE_VALUEB'variables/79/.ATTRIBUTES/VARIABLE_VALUEB'variables/80/.ATTRIBUTES/VARIABLE_VALUEB'variables/81/.ATTRIBUTES/VARIABLE_VALUEB'variables/82/.ATTRIBUTES/VARIABLE_VALUEB'variables/83/.ATTRIBUTES/VARIABLE_VALUEB'variables/84/.ATTRIBUTES/VARIABLE_VALUEB'variables/85/.ATTRIBUTES/VARIABLE_VALUEB'variables/86/.ATTRIBUTES/VARIABLE_VALUEB'variables/87/.ATTRIBUTES/VARIABLE_VALUEB'variables/88/.ATTRIBUTES/VARIABLE_VALUEB'variables/89/.ATTRIBUTES/VARIABLE_VALUEB'variables/90/.ATTRIBUTES/VARIABLE_VALUEB'variables/91/.ATTRIBUTES/VARIABLE_VALUEB'variables/92/.ATTRIBUTES/VARIABLE_VALUEB'variables/93/.ATTRIBUTES/VARIABLE_VALUEB'variables/94/.ATTRIBUTES/VARIABLE_VALUEB'variables/95/.ATTRIBUTES/VARIABLE_VALUEB'variables/96/.ATTRIBUTES/VARIABLE_VALUEB'variables/97/.ATTRIBUTES/VARIABLE_VALUEB'variables/98/.ATTRIBUTES/VARIABLE_VALUEB'variables/99/.ATTRIBUTES/VARIABLE_VALUEB(variables/100/.ATTRIBUTES/VARIABLE_VALUEB(variables/101/.ATTRIBUTES/VARIABLE_VALUEB(variables/102/.ATTRIBUTES/VARIABLE_VALUEB(variables/103/.ATTRIBUTES/VARIABLE_VALUEB(variables/104/.ATTRIBUTES/VARIABLE_VALUEB(variables/105/.ATTRIBUTES/VARIABLE_VALUEB(variables/106/.ATTRIBUTES/VARIABLE_VALUEB(variables/107/.ATTRIBUTES/VARIABLE_VALUEB(variables/108/.ATTRIBUTES/VARIABLE_VALUEB(variables/109/.ATTRIBUTES/VARIABLE_VALUEB(variables/110/.ATTRIBUTES/VARIABLE_VALUEB(variables/111/.ATTRIBUTES/VARIABLE_VALUEB(variables/112/.ATTRIBUTES/VARIABLE_VALUEB(variables/113/.ATTRIBUTES/VARIABLE_VALUEB(variables/114/.ATTRIBUTES/VARIABLE_VALUEB(variables/115/.ATTRIBUTES/VARIABLE_VALUEB(variables/116/.ATTRIBUTES/VARIABLE_VALUEB(variables/117/.ATTRIBUTES/VARIABLE_VALUEB(variables/118/.ATTRIBUTES/VARIABLE_VALUEB(variables/119/.ATTRIBUTES/VARIABLE_VALUEB(variables/120/.ATTRIBUTES/VARIABLE_VALUEB(variables/121/.ATTRIBUTES/VARIABLE_VALUEB(variables/122/.ATTRIBUTES/VARIABLE_VALUEB(variables/123/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/29/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/30/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/31/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/32/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/33/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/34/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/35/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/36/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/37/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/38/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/39/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/40/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/41/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/42/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/43/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/44/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/45/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/46/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/47/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/48/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/49/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/50/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/51/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/52/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/53/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/54/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/55/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/56/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/57/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/58/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/59/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/60/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/61/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/62/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/63/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/64/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/65/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/66/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/67/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/68/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/69/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/70/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/71/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/72/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/73/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/74/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/75/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/76/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/77/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/78/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/79/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/80/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/81/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/82/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/83/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/84/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/85/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/86/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/87/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/88/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/89/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/90/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/91/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/92/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/93/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/94/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/95/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/96/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/97/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/98/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/99/.ATTRIBUTES/VARIABLE_VALUEB-_all_variables/100/.ATTRIBUTES/VARIABLE_VALUEB-_all_variables/101/.ATTRIBUTES/VARIABLE_VALUEB-_all_variables/102/.ATTRIBUTES/VARIABLE_VALUEB-_all_variables/103/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �,
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0Identity_211:output:0Identity_213:output:0Identity_215:output:0Identity_217:output:0Identity_219:output:0Identity_221:output:0Identity_223:output:0Identity_225:output:0Identity_227:output:0Identity_229:output:0Identity_231:output:0Identity_233:output:0Identity_235:output:0Identity_237:output:0Identity_239:output:0Identity_241:output:0Identity_243:output:0Identity_245:output:0Identity_247:output:0Identity_249:output:0Identity_251:output:0Identity_253:output:0Identity_255:output:0Identity_257:output:0Identity_259:output:0Identity_261:output:0Identity_263:output:0Identity_265:output:0Identity_267:output:0Identity_269:output:0Identity_271:output:0Identity_273:output:0Identity_275:output:0Identity_277:output:0Identity_279:output:0Identity_281:output:0Identity_283:output:0Identity_285:output:0Identity_287:output:0Identity_289:output:0Identity_291:output:0Identity_293:output:0Identity_295:output:0Identity_297:output:0Identity_299:output:0Identity_301:output:0Identity_303:output:0Identity_305:output:0Identity_307:output:0Identity_309:output:0Identity_311:output:0Identity_313:output:0Identity_315:output:0Identity_317:output:0Identity_319:output:0Identity_321:output:0Identity_323:output:0Identity_325:output:0Identity_327:output:0Identity_329:output:0Identity_331:output:0Identity_333:output:0Identity_335:output:0Identity_337:output:0Identity_339:output:0Identity_341:output:0Identity_343:output:0Identity_345:output:0Identity_347:output:0Identity_349:output:0Identity_351:output:0Identity_353:output:0Identity_355:output:0Identity_357:output:0Identity_359:output:0Identity_361:output:0Identity_363:output:0Identity_365:output:0Identity_367:output:0Identity_369:output:0Identity_371:output:0Identity_373:output:0Identity_375:output:0Identity_377:output:0Identity_379:output:0Identity_381:output:0Identity_383:output:0Identity_385:output:0Identity_387:output:0Identity_389:output:0Identity_391:output:0Identity_393:output:0Identity_395:output:0Identity_397:output:0Identity_399:output:0Identity_401:output:0Identity_403:output:0Identity_405:output:0Identity_407:output:0Identity_409:output:0Identity_411:output:0Identity_413:output:0Identity_415:output:0Identity_417:output:0Identity_419:output:0Identity_421:output:0Identity_423:output:0Identity_425:output:0Identity_427:output:0Identity_429:output:0Identity_431:output:0Identity_433:output:0Identity_435:output:0Identity_437:output:0Identity_439:output:0Identity_441:output:0Identity_443:output:0Identity_445:output:0Identity_447:output:0Identity_449:output:0Identity_451:output:0Identity_453:output:0Identity_455:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *�
dtypes�
�2��
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_456Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_457IdentityIdentity_456:output:0^NoOp*
T0*
_output_shapes
: �`
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_105/DisableCopyOnRead^Read_105/ReadVariableOp^Read_106/DisableCopyOnRead^Read_106/ReadVariableOp^Read_107/DisableCopyOnRead^Read_107/ReadVariableOp^Read_108/DisableCopyOnRead^Read_108/ReadVariableOp^Read_109/DisableCopyOnRead^Read_109/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_110/DisableCopyOnRead^Read_110/ReadVariableOp^Read_111/DisableCopyOnRead^Read_111/ReadVariableOp^Read_112/DisableCopyOnRead^Read_112/ReadVariableOp^Read_113/DisableCopyOnRead^Read_113/ReadVariableOp^Read_114/DisableCopyOnRead^Read_114/ReadVariableOp^Read_115/DisableCopyOnRead^Read_115/ReadVariableOp^Read_116/DisableCopyOnRead^Read_116/ReadVariableOp^Read_117/DisableCopyOnRead^Read_117/ReadVariableOp^Read_118/DisableCopyOnRead^Read_118/ReadVariableOp^Read_119/DisableCopyOnRead^Read_119/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_120/DisableCopyOnRead^Read_120/ReadVariableOp^Read_121/DisableCopyOnRead^Read_121/ReadVariableOp^Read_122/DisableCopyOnRead^Read_122/ReadVariableOp^Read_123/DisableCopyOnRead^Read_123/ReadVariableOp^Read_124/DisableCopyOnRead^Read_124/ReadVariableOp^Read_125/DisableCopyOnRead^Read_125/ReadVariableOp^Read_126/DisableCopyOnRead^Read_126/ReadVariableOp^Read_127/DisableCopyOnRead^Read_127/ReadVariableOp^Read_128/DisableCopyOnRead^Read_128/ReadVariableOp^Read_129/DisableCopyOnRead^Read_129/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_130/DisableCopyOnRead^Read_130/ReadVariableOp^Read_131/DisableCopyOnRead^Read_131/ReadVariableOp^Read_132/DisableCopyOnRead^Read_132/ReadVariableOp^Read_133/DisableCopyOnRead^Read_133/ReadVariableOp^Read_134/DisableCopyOnRead^Read_134/ReadVariableOp^Read_135/DisableCopyOnRead^Read_135/ReadVariableOp^Read_136/DisableCopyOnRead^Read_136/ReadVariableOp^Read_137/DisableCopyOnRead^Read_137/ReadVariableOp^Read_138/DisableCopyOnRead^Read_138/ReadVariableOp^Read_139/DisableCopyOnRead^Read_139/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_140/DisableCopyOnRead^Read_140/ReadVariableOp^Read_141/DisableCopyOnRead^Read_141/ReadVariableOp^Read_142/DisableCopyOnRead^Read_142/ReadVariableOp^Read_143/DisableCopyOnRead^Read_143/ReadVariableOp^Read_144/DisableCopyOnRead^Read_144/ReadVariableOp^Read_145/DisableCopyOnRead^Read_145/ReadVariableOp^Read_146/DisableCopyOnRead^Read_146/ReadVariableOp^Read_147/DisableCopyOnRead^Read_147/ReadVariableOp^Read_148/DisableCopyOnRead^Read_148/ReadVariableOp^Read_149/DisableCopyOnRead^Read_149/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_150/DisableCopyOnRead^Read_150/ReadVariableOp^Read_151/DisableCopyOnRead^Read_151/ReadVariableOp^Read_152/DisableCopyOnRead^Read_152/ReadVariableOp^Read_153/DisableCopyOnRead^Read_153/ReadVariableOp^Read_154/DisableCopyOnRead^Read_154/ReadVariableOp^Read_155/DisableCopyOnRead^Read_155/ReadVariableOp^Read_156/DisableCopyOnRead^Read_156/ReadVariableOp^Read_157/DisableCopyOnRead^Read_157/ReadVariableOp^Read_158/DisableCopyOnRead^Read_158/ReadVariableOp^Read_159/DisableCopyOnRead^Read_159/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_160/DisableCopyOnRead^Read_160/ReadVariableOp^Read_161/DisableCopyOnRead^Read_161/ReadVariableOp^Read_162/DisableCopyOnRead^Read_162/ReadVariableOp^Read_163/DisableCopyOnRead^Read_163/ReadVariableOp^Read_164/DisableCopyOnRead^Read_164/ReadVariableOp^Read_165/DisableCopyOnRead^Read_165/ReadVariableOp^Read_166/DisableCopyOnRead^Read_166/ReadVariableOp^Read_167/DisableCopyOnRead^Read_167/ReadVariableOp^Read_168/DisableCopyOnRead^Read_168/ReadVariableOp^Read_169/DisableCopyOnRead^Read_169/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_170/DisableCopyOnRead^Read_170/ReadVariableOp^Read_171/DisableCopyOnRead^Read_171/ReadVariableOp^Read_172/DisableCopyOnRead^Read_172/ReadVariableOp^Read_173/DisableCopyOnRead^Read_173/ReadVariableOp^Read_174/DisableCopyOnRead^Read_174/ReadVariableOp^Read_175/DisableCopyOnRead^Read_175/ReadVariableOp^Read_176/DisableCopyOnRead^Read_176/ReadVariableOp^Read_177/DisableCopyOnRead^Read_177/ReadVariableOp^Read_178/DisableCopyOnRead^Read_178/ReadVariableOp^Read_179/DisableCopyOnRead^Read_179/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_180/DisableCopyOnRead^Read_180/ReadVariableOp^Read_181/DisableCopyOnRead^Read_181/ReadVariableOp^Read_182/DisableCopyOnRead^Read_182/ReadVariableOp^Read_183/DisableCopyOnRead^Read_183/ReadVariableOp^Read_184/DisableCopyOnRead^Read_184/ReadVariableOp^Read_185/DisableCopyOnRead^Read_185/ReadVariableOp^Read_186/DisableCopyOnRead^Read_186/ReadVariableOp^Read_187/DisableCopyOnRead^Read_187/ReadVariableOp^Read_188/DisableCopyOnRead^Read_188/ReadVariableOp^Read_189/DisableCopyOnRead^Read_189/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_190/DisableCopyOnRead^Read_190/ReadVariableOp^Read_191/DisableCopyOnRead^Read_191/ReadVariableOp^Read_192/DisableCopyOnRead^Read_192/ReadVariableOp^Read_193/DisableCopyOnRead^Read_193/ReadVariableOp^Read_194/DisableCopyOnRead^Read_194/ReadVariableOp^Read_195/DisableCopyOnRead^Read_195/ReadVariableOp^Read_196/DisableCopyOnRead^Read_196/ReadVariableOp^Read_197/DisableCopyOnRead^Read_197/ReadVariableOp^Read_198/DisableCopyOnRead^Read_198/ReadVariableOp^Read_199/DisableCopyOnRead^Read_199/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_200/DisableCopyOnRead^Read_200/ReadVariableOp^Read_201/DisableCopyOnRead^Read_201/ReadVariableOp^Read_202/DisableCopyOnRead^Read_202/ReadVariableOp^Read_203/DisableCopyOnRead^Read_203/ReadVariableOp^Read_204/DisableCopyOnRead^Read_204/ReadVariableOp^Read_205/DisableCopyOnRead^Read_205/ReadVariableOp^Read_206/DisableCopyOnRead^Read_206/ReadVariableOp^Read_207/DisableCopyOnRead^Read_207/ReadVariableOp^Read_208/DisableCopyOnRead^Read_208/ReadVariableOp^Read_209/DisableCopyOnRead^Read_209/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_210/DisableCopyOnRead^Read_210/ReadVariableOp^Read_211/DisableCopyOnRead^Read_211/ReadVariableOp^Read_212/DisableCopyOnRead^Read_212/ReadVariableOp^Read_213/DisableCopyOnRead^Read_213/ReadVariableOp^Read_214/DisableCopyOnRead^Read_214/ReadVariableOp^Read_215/DisableCopyOnRead^Read_215/ReadVariableOp^Read_216/DisableCopyOnRead^Read_216/ReadVariableOp^Read_217/DisableCopyOnRead^Read_217/ReadVariableOp^Read_218/DisableCopyOnRead^Read_218/ReadVariableOp^Read_219/DisableCopyOnRead^Read_219/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_220/DisableCopyOnRead^Read_220/ReadVariableOp^Read_221/DisableCopyOnRead^Read_221/ReadVariableOp^Read_222/DisableCopyOnRead^Read_222/ReadVariableOp^Read_223/DisableCopyOnRead^Read_223/ReadVariableOp^Read_224/DisableCopyOnRead^Read_224/ReadVariableOp^Read_225/DisableCopyOnRead^Read_225/ReadVariableOp^Read_226/DisableCopyOnRead^Read_226/ReadVariableOp^Read_227/DisableCopyOnRead^Read_227/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_457Identity_457:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp28
Read_105/DisableCopyOnReadRead_105/DisableCopyOnRead22
Read_105/ReadVariableOpRead_105/ReadVariableOp28
Read_106/DisableCopyOnReadRead_106/DisableCopyOnRead22
Read_106/ReadVariableOpRead_106/ReadVariableOp28
Read_107/DisableCopyOnReadRead_107/DisableCopyOnRead22
Read_107/ReadVariableOpRead_107/ReadVariableOp28
Read_108/DisableCopyOnReadRead_108/DisableCopyOnRead22
Read_108/ReadVariableOpRead_108/ReadVariableOp28
Read_109/DisableCopyOnReadRead_109/DisableCopyOnRead22
Read_109/ReadVariableOpRead_109/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp28
Read_110/DisableCopyOnReadRead_110/DisableCopyOnRead22
Read_110/ReadVariableOpRead_110/ReadVariableOp28
Read_111/DisableCopyOnReadRead_111/DisableCopyOnRead22
Read_111/ReadVariableOpRead_111/ReadVariableOp28
Read_112/DisableCopyOnReadRead_112/DisableCopyOnRead22
Read_112/ReadVariableOpRead_112/ReadVariableOp28
Read_113/DisableCopyOnReadRead_113/DisableCopyOnRead22
Read_113/ReadVariableOpRead_113/ReadVariableOp28
Read_114/DisableCopyOnReadRead_114/DisableCopyOnRead22
Read_114/ReadVariableOpRead_114/ReadVariableOp28
Read_115/DisableCopyOnReadRead_115/DisableCopyOnRead22
Read_115/ReadVariableOpRead_115/ReadVariableOp28
Read_116/DisableCopyOnReadRead_116/DisableCopyOnRead22
Read_116/ReadVariableOpRead_116/ReadVariableOp28
Read_117/DisableCopyOnReadRead_117/DisableCopyOnRead22
Read_117/ReadVariableOpRead_117/ReadVariableOp28
Read_118/DisableCopyOnReadRead_118/DisableCopyOnRead22
Read_118/ReadVariableOpRead_118/ReadVariableOp28
Read_119/DisableCopyOnReadRead_119/DisableCopyOnRead22
Read_119/ReadVariableOpRead_119/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp28
Read_120/DisableCopyOnReadRead_120/DisableCopyOnRead22
Read_120/ReadVariableOpRead_120/ReadVariableOp28
Read_121/DisableCopyOnReadRead_121/DisableCopyOnRead22
Read_121/ReadVariableOpRead_121/ReadVariableOp28
Read_122/DisableCopyOnReadRead_122/DisableCopyOnRead22
Read_122/ReadVariableOpRead_122/ReadVariableOp28
Read_123/DisableCopyOnReadRead_123/DisableCopyOnRead22
Read_123/ReadVariableOpRead_123/ReadVariableOp28
Read_124/DisableCopyOnReadRead_124/DisableCopyOnRead22
Read_124/ReadVariableOpRead_124/ReadVariableOp28
Read_125/DisableCopyOnReadRead_125/DisableCopyOnRead22
Read_125/ReadVariableOpRead_125/ReadVariableOp28
Read_126/DisableCopyOnReadRead_126/DisableCopyOnRead22
Read_126/ReadVariableOpRead_126/ReadVariableOp28
Read_127/DisableCopyOnReadRead_127/DisableCopyOnRead22
Read_127/ReadVariableOpRead_127/ReadVariableOp28
Read_128/DisableCopyOnReadRead_128/DisableCopyOnRead22
Read_128/ReadVariableOpRead_128/ReadVariableOp28
Read_129/DisableCopyOnReadRead_129/DisableCopyOnRead22
Read_129/ReadVariableOpRead_129/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp28
Read_130/DisableCopyOnReadRead_130/DisableCopyOnRead22
Read_130/ReadVariableOpRead_130/ReadVariableOp28
Read_131/DisableCopyOnReadRead_131/DisableCopyOnRead22
Read_131/ReadVariableOpRead_131/ReadVariableOp28
Read_132/DisableCopyOnReadRead_132/DisableCopyOnRead22
Read_132/ReadVariableOpRead_132/ReadVariableOp28
Read_133/DisableCopyOnReadRead_133/DisableCopyOnRead22
Read_133/ReadVariableOpRead_133/ReadVariableOp28
Read_134/DisableCopyOnReadRead_134/DisableCopyOnRead22
Read_134/ReadVariableOpRead_134/ReadVariableOp28
Read_135/DisableCopyOnReadRead_135/DisableCopyOnRead22
Read_135/ReadVariableOpRead_135/ReadVariableOp28
Read_136/DisableCopyOnReadRead_136/DisableCopyOnRead22
Read_136/ReadVariableOpRead_136/ReadVariableOp28
Read_137/DisableCopyOnReadRead_137/DisableCopyOnRead22
Read_137/ReadVariableOpRead_137/ReadVariableOp28
Read_138/DisableCopyOnReadRead_138/DisableCopyOnRead22
Read_138/ReadVariableOpRead_138/ReadVariableOp28
Read_139/DisableCopyOnReadRead_139/DisableCopyOnRead22
Read_139/ReadVariableOpRead_139/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp28
Read_140/DisableCopyOnReadRead_140/DisableCopyOnRead22
Read_140/ReadVariableOpRead_140/ReadVariableOp28
Read_141/DisableCopyOnReadRead_141/DisableCopyOnRead22
Read_141/ReadVariableOpRead_141/ReadVariableOp28
Read_142/DisableCopyOnReadRead_142/DisableCopyOnRead22
Read_142/ReadVariableOpRead_142/ReadVariableOp28
Read_143/DisableCopyOnReadRead_143/DisableCopyOnRead22
Read_143/ReadVariableOpRead_143/ReadVariableOp28
Read_144/DisableCopyOnReadRead_144/DisableCopyOnRead22
Read_144/ReadVariableOpRead_144/ReadVariableOp28
Read_145/DisableCopyOnReadRead_145/DisableCopyOnRead22
Read_145/ReadVariableOpRead_145/ReadVariableOp28
Read_146/DisableCopyOnReadRead_146/DisableCopyOnRead22
Read_146/ReadVariableOpRead_146/ReadVariableOp28
Read_147/DisableCopyOnReadRead_147/DisableCopyOnRead22
Read_147/ReadVariableOpRead_147/ReadVariableOp28
Read_148/DisableCopyOnReadRead_148/DisableCopyOnRead22
Read_148/ReadVariableOpRead_148/ReadVariableOp28
Read_149/DisableCopyOnReadRead_149/DisableCopyOnRead22
Read_149/ReadVariableOpRead_149/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp28
Read_150/DisableCopyOnReadRead_150/DisableCopyOnRead22
Read_150/ReadVariableOpRead_150/ReadVariableOp28
Read_151/DisableCopyOnReadRead_151/DisableCopyOnRead22
Read_151/ReadVariableOpRead_151/ReadVariableOp28
Read_152/DisableCopyOnReadRead_152/DisableCopyOnRead22
Read_152/ReadVariableOpRead_152/ReadVariableOp28
Read_153/DisableCopyOnReadRead_153/DisableCopyOnRead22
Read_153/ReadVariableOpRead_153/ReadVariableOp28
Read_154/DisableCopyOnReadRead_154/DisableCopyOnRead22
Read_154/ReadVariableOpRead_154/ReadVariableOp28
Read_155/DisableCopyOnReadRead_155/DisableCopyOnRead22
Read_155/ReadVariableOpRead_155/ReadVariableOp28
Read_156/DisableCopyOnReadRead_156/DisableCopyOnRead22
Read_156/ReadVariableOpRead_156/ReadVariableOp28
Read_157/DisableCopyOnReadRead_157/DisableCopyOnRead22
Read_157/ReadVariableOpRead_157/ReadVariableOp28
Read_158/DisableCopyOnReadRead_158/DisableCopyOnRead22
Read_158/ReadVariableOpRead_158/ReadVariableOp28
Read_159/DisableCopyOnReadRead_159/DisableCopyOnRead22
Read_159/ReadVariableOpRead_159/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp28
Read_160/DisableCopyOnReadRead_160/DisableCopyOnRead22
Read_160/ReadVariableOpRead_160/ReadVariableOp28
Read_161/DisableCopyOnReadRead_161/DisableCopyOnRead22
Read_161/ReadVariableOpRead_161/ReadVariableOp28
Read_162/DisableCopyOnReadRead_162/DisableCopyOnRead22
Read_162/ReadVariableOpRead_162/ReadVariableOp28
Read_163/DisableCopyOnReadRead_163/DisableCopyOnRead22
Read_163/ReadVariableOpRead_163/ReadVariableOp28
Read_164/DisableCopyOnReadRead_164/DisableCopyOnRead22
Read_164/ReadVariableOpRead_164/ReadVariableOp28
Read_165/DisableCopyOnReadRead_165/DisableCopyOnRead22
Read_165/ReadVariableOpRead_165/ReadVariableOp28
Read_166/DisableCopyOnReadRead_166/DisableCopyOnRead22
Read_166/ReadVariableOpRead_166/ReadVariableOp28
Read_167/DisableCopyOnReadRead_167/DisableCopyOnRead22
Read_167/ReadVariableOpRead_167/ReadVariableOp28
Read_168/DisableCopyOnReadRead_168/DisableCopyOnRead22
Read_168/ReadVariableOpRead_168/ReadVariableOp28
Read_169/DisableCopyOnReadRead_169/DisableCopyOnRead22
Read_169/ReadVariableOpRead_169/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp28
Read_170/DisableCopyOnReadRead_170/DisableCopyOnRead22
Read_170/ReadVariableOpRead_170/ReadVariableOp28
Read_171/DisableCopyOnReadRead_171/DisableCopyOnRead22
Read_171/ReadVariableOpRead_171/ReadVariableOp28
Read_172/DisableCopyOnReadRead_172/DisableCopyOnRead22
Read_172/ReadVariableOpRead_172/ReadVariableOp28
Read_173/DisableCopyOnReadRead_173/DisableCopyOnRead22
Read_173/ReadVariableOpRead_173/ReadVariableOp28
Read_174/DisableCopyOnReadRead_174/DisableCopyOnRead22
Read_174/ReadVariableOpRead_174/ReadVariableOp28
Read_175/DisableCopyOnReadRead_175/DisableCopyOnRead22
Read_175/ReadVariableOpRead_175/ReadVariableOp28
Read_176/DisableCopyOnReadRead_176/DisableCopyOnRead22
Read_176/ReadVariableOpRead_176/ReadVariableOp28
Read_177/DisableCopyOnReadRead_177/DisableCopyOnRead22
Read_177/ReadVariableOpRead_177/ReadVariableOp28
Read_178/DisableCopyOnReadRead_178/DisableCopyOnRead22
Read_178/ReadVariableOpRead_178/ReadVariableOp28
Read_179/DisableCopyOnReadRead_179/DisableCopyOnRead22
Read_179/ReadVariableOpRead_179/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp28
Read_180/DisableCopyOnReadRead_180/DisableCopyOnRead22
Read_180/ReadVariableOpRead_180/ReadVariableOp28
Read_181/DisableCopyOnReadRead_181/DisableCopyOnRead22
Read_181/ReadVariableOpRead_181/ReadVariableOp28
Read_182/DisableCopyOnReadRead_182/DisableCopyOnRead22
Read_182/ReadVariableOpRead_182/ReadVariableOp28
Read_183/DisableCopyOnReadRead_183/DisableCopyOnRead22
Read_183/ReadVariableOpRead_183/ReadVariableOp28
Read_184/DisableCopyOnReadRead_184/DisableCopyOnRead22
Read_184/ReadVariableOpRead_184/ReadVariableOp28
Read_185/DisableCopyOnReadRead_185/DisableCopyOnRead22
Read_185/ReadVariableOpRead_185/ReadVariableOp28
Read_186/DisableCopyOnReadRead_186/DisableCopyOnRead22
Read_186/ReadVariableOpRead_186/ReadVariableOp28
Read_187/DisableCopyOnReadRead_187/DisableCopyOnRead22
Read_187/ReadVariableOpRead_187/ReadVariableOp28
Read_188/DisableCopyOnReadRead_188/DisableCopyOnRead22
Read_188/ReadVariableOpRead_188/ReadVariableOp28
Read_189/DisableCopyOnReadRead_189/DisableCopyOnRead22
Read_189/ReadVariableOpRead_189/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp28
Read_190/DisableCopyOnReadRead_190/DisableCopyOnRead22
Read_190/ReadVariableOpRead_190/ReadVariableOp28
Read_191/DisableCopyOnReadRead_191/DisableCopyOnRead22
Read_191/ReadVariableOpRead_191/ReadVariableOp28
Read_192/DisableCopyOnReadRead_192/DisableCopyOnRead22
Read_192/ReadVariableOpRead_192/ReadVariableOp28
Read_193/DisableCopyOnReadRead_193/DisableCopyOnRead22
Read_193/ReadVariableOpRead_193/ReadVariableOp28
Read_194/DisableCopyOnReadRead_194/DisableCopyOnRead22
Read_194/ReadVariableOpRead_194/ReadVariableOp28
Read_195/DisableCopyOnReadRead_195/DisableCopyOnRead22
Read_195/ReadVariableOpRead_195/ReadVariableOp28
Read_196/DisableCopyOnReadRead_196/DisableCopyOnRead22
Read_196/ReadVariableOpRead_196/ReadVariableOp28
Read_197/DisableCopyOnReadRead_197/DisableCopyOnRead22
Read_197/ReadVariableOpRead_197/ReadVariableOp28
Read_198/DisableCopyOnReadRead_198/DisableCopyOnRead22
Read_198/ReadVariableOpRead_198/ReadVariableOp28
Read_199/DisableCopyOnReadRead_199/DisableCopyOnRead22
Read_199/ReadVariableOpRead_199/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp28
Read_200/DisableCopyOnReadRead_200/DisableCopyOnRead22
Read_200/ReadVariableOpRead_200/ReadVariableOp28
Read_201/DisableCopyOnReadRead_201/DisableCopyOnRead22
Read_201/ReadVariableOpRead_201/ReadVariableOp28
Read_202/DisableCopyOnReadRead_202/DisableCopyOnRead22
Read_202/ReadVariableOpRead_202/ReadVariableOp28
Read_203/DisableCopyOnReadRead_203/DisableCopyOnRead22
Read_203/ReadVariableOpRead_203/ReadVariableOp28
Read_204/DisableCopyOnReadRead_204/DisableCopyOnRead22
Read_204/ReadVariableOpRead_204/ReadVariableOp28
Read_205/DisableCopyOnReadRead_205/DisableCopyOnRead22
Read_205/ReadVariableOpRead_205/ReadVariableOp28
Read_206/DisableCopyOnReadRead_206/DisableCopyOnRead22
Read_206/ReadVariableOpRead_206/ReadVariableOp28
Read_207/DisableCopyOnReadRead_207/DisableCopyOnRead22
Read_207/ReadVariableOpRead_207/ReadVariableOp28
Read_208/DisableCopyOnReadRead_208/DisableCopyOnRead22
Read_208/ReadVariableOpRead_208/ReadVariableOp28
Read_209/DisableCopyOnReadRead_209/DisableCopyOnRead22
Read_209/ReadVariableOpRead_209/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp28
Read_210/DisableCopyOnReadRead_210/DisableCopyOnRead22
Read_210/ReadVariableOpRead_210/ReadVariableOp28
Read_211/DisableCopyOnReadRead_211/DisableCopyOnRead22
Read_211/ReadVariableOpRead_211/ReadVariableOp28
Read_212/DisableCopyOnReadRead_212/DisableCopyOnRead22
Read_212/ReadVariableOpRead_212/ReadVariableOp28
Read_213/DisableCopyOnReadRead_213/DisableCopyOnRead22
Read_213/ReadVariableOpRead_213/ReadVariableOp28
Read_214/DisableCopyOnReadRead_214/DisableCopyOnRead22
Read_214/ReadVariableOpRead_214/ReadVariableOp28
Read_215/DisableCopyOnReadRead_215/DisableCopyOnRead22
Read_215/ReadVariableOpRead_215/ReadVariableOp28
Read_216/DisableCopyOnReadRead_216/DisableCopyOnRead22
Read_216/ReadVariableOpRead_216/ReadVariableOp28
Read_217/DisableCopyOnReadRead_217/DisableCopyOnRead22
Read_217/ReadVariableOpRead_217/ReadVariableOp28
Read_218/DisableCopyOnReadRead_218/DisableCopyOnRead22
Read_218/ReadVariableOpRead_218/ReadVariableOp28
Read_219/DisableCopyOnReadRead_219/DisableCopyOnRead22
Read_219/ReadVariableOpRead_219/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp28
Read_220/DisableCopyOnReadRead_220/DisableCopyOnRead22
Read_220/ReadVariableOpRead_220/ReadVariableOp28
Read_221/DisableCopyOnReadRead_221/DisableCopyOnRead22
Read_221/ReadVariableOpRead_221/ReadVariableOp28
Read_222/DisableCopyOnReadRead_222/DisableCopyOnRead22
Read_222/ReadVariableOpRead_222/ReadVariableOp28
Read_223/DisableCopyOnReadRead_223/DisableCopyOnRead22
Read_223/ReadVariableOpRead_223/ReadVariableOp28
Read_224/DisableCopyOnReadRead_224/DisableCopyOnRead22
Read_224/ReadVariableOpRead_224/ReadVariableOp28
Read_225/DisableCopyOnReadRead_225/DisableCopyOnRead22
Read_225/ReadVariableOpRead_225/ReadVariableOp28
Read_226/DisableCopyOnReadRead_226/DisableCopyOnRead22
Read_226/ReadVariableOpRead_226/ReadVariableOp28
Read_227/DisableCopyOnReadRead_227/DisableCopyOnRead22
Read_227/ReadVariableOpRead_227/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_user_specified_nameVariable_123:,(
&
_user_specified_nameVariable_122:,(
&
_user_specified_nameVariable_121:,(
&
_user_specified_nameVariable_120:,(
&
_user_specified_nameVariable_119:,(
&
_user_specified_nameVariable_118:,(
&
_user_specified_nameVariable_117:,(
&
_user_specified_nameVariable_116:,	(
&
_user_specified_nameVariable_115:,
(
&
_user_specified_nameVariable_114:,(
&
_user_specified_nameVariable_113:,(
&
_user_specified_nameVariable_112:,(
&
_user_specified_nameVariable_111:,(
&
_user_specified_nameVariable_110:,(
&
_user_specified_nameVariable_109:,(
&
_user_specified_nameVariable_108:,(
&
_user_specified_nameVariable_107:,(
&
_user_specified_nameVariable_106:,(
&
_user_specified_nameVariable_105:,(
&
_user_specified_nameVariable_104:,(
&
_user_specified_nameVariable_103:,(
&
_user_specified_nameVariable_102:,(
&
_user_specified_nameVariable_101:,(
&
_user_specified_nameVariable_100:+'
%
_user_specified_nameVariable_99:+'
%
_user_specified_nameVariable_98:+'
%
_user_specified_nameVariable_97:+'
%
_user_specified_nameVariable_96:+'
%
_user_specified_nameVariable_95:+'
%
_user_specified_nameVariable_94:+'
%
_user_specified_nameVariable_93:+ '
%
_user_specified_nameVariable_92:+!'
%
_user_specified_nameVariable_91:+"'
%
_user_specified_nameVariable_90:+#'
%
_user_specified_nameVariable_89:+$'
%
_user_specified_nameVariable_88:+%'
%
_user_specified_nameVariable_87:+&'
%
_user_specified_nameVariable_86:+''
%
_user_specified_nameVariable_85:+('
%
_user_specified_nameVariable_84:+)'
%
_user_specified_nameVariable_83:+*'
%
_user_specified_nameVariable_82:++'
%
_user_specified_nameVariable_81:+,'
%
_user_specified_nameVariable_80:+-'
%
_user_specified_nameVariable_79:+.'
%
_user_specified_nameVariable_78:+/'
%
_user_specified_nameVariable_77:+0'
%
_user_specified_nameVariable_76:+1'
%
_user_specified_nameVariable_75:+2'
%
_user_specified_nameVariable_74:+3'
%
_user_specified_nameVariable_73:+4'
%
_user_specified_nameVariable_72:+5'
%
_user_specified_nameVariable_71:+6'
%
_user_specified_nameVariable_70:+7'
%
_user_specified_nameVariable_69:+8'
%
_user_specified_nameVariable_68:+9'
%
_user_specified_nameVariable_67:+:'
%
_user_specified_nameVariable_66:+;'
%
_user_specified_nameVariable_65:+<'
%
_user_specified_nameVariable_64:+='
%
_user_specified_nameVariable_63:+>'
%
_user_specified_nameVariable_62:+?'
%
_user_specified_nameVariable_61:+@'
%
_user_specified_nameVariable_60:+A'
%
_user_specified_nameVariable_59:+B'
%
_user_specified_nameVariable_58:+C'
%
_user_specified_nameVariable_57:+D'
%
_user_specified_nameVariable_56:+E'
%
_user_specified_nameVariable_55:+F'
%
_user_specified_nameVariable_54:+G'
%
_user_specified_nameVariable_53:+H'
%
_user_specified_nameVariable_52:+I'
%
_user_specified_nameVariable_51:+J'
%
_user_specified_nameVariable_50:+K'
%
_user_specified_nameVariable_49:+L'
%
_user_specified_nameVariable_48:+M'
%
_user_specified_nameVariable_47:+N'
%
_user_specified_nameVariable_46:+O'
%
_user_specified_nameVariable_45:+P'
%
_user_specified_nameVariable_44:+Q'
%
_user_specified_nameVariable_43:+R'
%
_user_specified_nameVariable_42:+S'
%
_user_specified_nameVariable_41:+T'
%
_user_specified_nameVariable_40:+U'
%
_user_specified_nameVariable_39:+V'
%
_user_specified_nameVariable_38:+W'
%
_user_specified_nameVariable_37:+X'
%
_user_specified_nameVariable_36:+Y'
%
_user_specified_nameVariable_35:+Z'
%
_user_specified_nameVariable_34:+['
%
_user_specified_nameVariable_33:+\'
%
_user_specified_nameVariable_32:+]'
%
_user_specified_nameVariable_31:+^'
%
_user_specified_nameVariable_30:+_'
%
_user_specified_nameVariable_29:+`'
%
_user_specified_nameVariable_28:+a'
%
_user_specified_nameVariable_27:+b'
%
_user_specified_nameVariable_26:+c'
%
_user_specified_nameVariable_25:+d'
%
_user_specified_nameVariable_24:+e'
%
_user_specified_nameVariable_23:+f'
%
_user_specified_nameVariable_22:+g'
%
_user_specified_nameVariable_21:+h'
%
_user_specified_nameVariable_20:+i'
%
_user_specified_nameVariable_19:+j'
%
_user_specified_nameVariable_18:+k'
%
_user_specified_nameVariable_17:+l'
%
_user_specified_nameVariable_16:+m'
%
_user_specified_nameVariable_15:+n'
%
_user_specified_nameVariable_14:+o'
%
_user_specified_nameVariable_13:+p'
%
_user_specified_nameVariable_12:+q'
%
_user_specified_nameVariable_11:+r'
%
_user_specified_nameVariable_10:*s&
$
_user_specified_name
Variable_9:*t&
$
_user_specified_name
Variable_8:*u&
$
_user_specified_name
Variable_7:*v&
$
_user_specified_name
Variable_6:*w&
$
_user_specified_name
Variable_5:*x&
$
_user_specified_name
Variable_4:*y&
$
_user_specified_name
Variable_3:*z&
$
_user_specified_name
Variable_2:*{&
$
_user_specified_name
Variable_1:(|$
"
_user_specified_name
Variable:W}S
Q
_user_specified_name97transformer_layer_3/self_attention_layer/query/kernel_1:U~Q
O
_user_specified_name75transformer_layer_3/self_attention_layer/value/bias_1:`\
Z
_user_specified_nameB@transformer_layer_3/self_attention_layer/attention_output/bias_1:T�O
M
_user_specified_name53transformer_layer_3/feedforward_output_dense/bias_1:U�P
N
_user_specified_name64transformer_layer_5/self_attention_layer_norm/beta_1:V�Q
O
_user_specified_name75transformer_layer_1/self_attention_layer/query/bias_1:V�Q
O
_user_specified_name75transformer_layer_3/self_attention_layer/key/kernel_1:U�P
N
_user_specified_name64transformer_layer_3/self_attention_layer_norm/beta_1:T�O
M
_user_specified_name53transformer_layer_1/self_attention_layer/key/bias_1:T�O
M
_user_specified_name53transformer_layer_4/feedforward_output_dense/bias_1:S�N
L
_user_specified_name42transformer_layer_4/feedforward_layer_norm/gamma_1:R�M
K
_user_specified_name31transformer_layer_0/feedforward_layer_norm/beta_1:V�Q
O
_user_specified_name75transformer_layer_4/self_attention_layer/query/bias_1:V�Q
O
_user_specified_name75transformer_layer_4/self_attention_layer/key/kernel_1:Z�U
S
_user_specified_name;9token_and_position_embedding/token_embedding/embeddings_1:T�O
M
_user_specified_name53transformer_layer_5/feedforward_output_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_3/self_attention_layer/query/bias_1:T�O
M
_user_specified_name53transformer_layer_3/self_attention_layer/key/bias_1:V�Q
O
_user_specified_name75transformer_layer_0/self_attention_layer_norm/gamma_1:R�M
K
_user_specified_name31transformer_layer_4/feedforward_layer_norm/beta_1:=�8
6
_user_specified_nameembeddings_layer_norm/beta_1:X�S
Q
_user_specified_name97transformer_layer_2/self_attention_layer/query/kernel_1:S�N
L
_user_specified_name42transformer_layer_2/feedforward_layer_norm/gamma_1:Z�U
S
_user_specified_name;9transformer_layer_4/feedforward_intermediate_dense/bias_1:T�O
M
_user_specified_name53transformer_layer_4/self_attention_layer/key/bias_1:X�S
Q
_user_specified_name97transformer_layer_4/self_attention_layer/value/kernel_1:V�Q
O
_user_specified_name75transformer_layer_1/feedforward_output_dense/kernel_1:c�^
\
_user_specified_nameDBtransformer_layer_0/self_attention_layer/attention_output/kernel_1:S�N
L
_user_specified_name42transformer_layer_1/feedforward_layer_norm/gamma_1:Z�U
S
_user_specified_name;9transformer_layer_1/feedforward_intermediate_dense/bias_1:4�/
-
_user_specified_namepooled_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_2/self_attention_layer/query/bias_1:R�M
K
_user_specified_name31transformer_layer_2/feedforward_layer_norm/beta_1:\�W
U
_user_specified_name=;transformer_layer_4/feedforward_intermediate_dense/kernel_1:c�^
\
_user_specified_nameDBtransformer_layer_4/self_attention_layer/attention_output/kernel_1:]�X
V
_user_specified_name><token_and_position_embedding/position_embedding/embeddings_1:V�Q
O
_user_specified_name75transformer_layer_2/self_attention_layer/key/kernel_1:Z�U
S
_user_specified_name;9transformer_layer_2/feedforward_intermediate_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_4/self_attention_layer/value/bias_1:S�N
L
_user_specified_name42transformer_layer_5/feedforward_layer_norm/gamma_1:0�+
)
_user_specified_namelogits/kernel_1:>�9
7
_user_specified_nameembeddings_layer_norm/gamma_1:X�S
Q
_user_specified_name97transformer_layer_2/self_attention_layer/value/kernel_1:S�N
L
_user_specified_name42transformer_layer_3/feedforward_layer_norm/gamma_1:V�Q
O
_user_specified_name75transformer_layer_4/self_attention_layer_norm/gamma_1:X�S
Q
_user_specified_name97transformer_layer_5/self_attention_layer/query/kernel_1:X�S
Q
_user_specified_name97transformer_layer_0/self_attention_layer/query/kernel_1:a�\
Z
_user_specified_nameB@transformer_layer_0/self_attention_layer/attention_output/bias_1:R�M
K
_user_specified_name31transformer_layer_1/feedforward_layer_norm/beta_1:V�Q
O
_user_specified_name75transformer_layer_5/self_attention_layer/key/kernel_1:V�Q
O
_user_specified_name75transformer_layer_0/self_attention_layer/key/kernel_1:U�P
N
_user_specified_name64transformer_layer_0/self_attention_layer_norm/beta_1:X�S
Q
_user_specified_name97transformer_layer_5/self_attention_layer/value/kernel_1:X�S
Q
_user_specified_name97transformer_layer_1/self_attention_layer/value/kernel_1:T�O
M
_user_specified_name53transformer_layer_2/self_attention_layer/key/bias_1:\�W
U
_user_specified_name=;transformer_layer_2/feedforward_intermediate_dense/kernel_1:a�\
Z
_user_specified_nameB@transformer_layer_4/self_attention_layer/attention_output/bias_1:R�M
K
_user_specified_name31transformer_layer_5/feedforward_layer_norm/beta_1:6�1
/
_user_specified_namepooled_dense/kernel_1:.�)
'
_user_specified_namelogits/bias_1:T�O
M
_user_specified_name53transformer_layer_0/feedforward_output_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_2/self_attention_layer/value/bias_1:c�^
\
_user_specified_nameDBtransformer_layer_2/self_attention_layer/attention_output/kernel_1:R�M
K
_user_specified_name31transformer_layer_3/feedforward_layer_norm/beta_1:U�P
N
_user_specified_name64transformer_layer_4/self_attention_layer_norm/beta_1:V�Q
O
_user_specified_name75transformer_layer_5/self_attention_layer/query/bias_1:\�W
U
_user_specified_name=;transformer_layer_5/feedforward_intermediate_dense/kernel_1:V�Q
O
_user_specified_name75transformer_layer_0/self_attention_layer/query/bias_1:V�Q
O
_user_specified_name75transformer_layer_2/self_attention_layer_norm/gamma_1:\�W
U
_user_specified_name=;transformer_layer_3/feedforward_intermediate_dense/kernel_1:T�O
M
_user_specified_name53transformer_layer_0/self_attention_layer/key/bias_1:\�W
U
_user_specified_name=;transformer_layer_1/feedforward_intermediate_dense/kernel_1:V�Q
O
_user_specified_name75transformer_layer_5/self_attention_layer/value/bias_1:c�^
\
_user_specified_nameDBtransformer_layer_5/self_attention_layer/attention_output/kernel_1:X�S
Q
_user_specified_name97transformer_layer_0/self_attention_layer/value/kernel_1:V�Q
O
_user_specified_name75transformer_layer_1/self_attention_layer/value/bias_1:c�^
\
_user_specified_nameDBtransformer_layer_1/self_attention_layer/attention_output/kernel_1:\�W
U
_user_specified_name=;transformer_layer_0/feedforward_intermediate_dense/kernel_1:V�Q
O
_user_specified_name75transformer_layer_1/self_attention_layer_norm/gamma_1:a�\
Z
_user_specified_nameB@transformer_layer_2/self_attention_layer/attention_output/bias_1:V�Q
O
_user_specified_name75transformer_layer_2/feedforward_output_dense/kernel_1:Z�U
S
_user_specified_name;9transformer_layer_5/feedforward_intermediate_dense/bias_1:U�P
N
_user_specified_name64transformer_layer_2/self_attention_layer_norm/beta_1:Z�U
S
_user_specified_name;9transformer_layer_3/feedforward_intermediate_dense/bias_1:T�O
M
_user_specified_name53transformer_layer_5/self_attention_layer/key/bias_1:T�O
M
_user_specified_name53transformer_layer_1/feedforward_output_dense/bias_1:X�S
Q
_user_specified_name97transformer_layer_3/self_attention_layer/value/kernel_1:c�^
\
_user_specified_nameDBtransformer_layer_3/self_attention_layer/attention_output/kernel_1:V�Q
O
_user_specified_name75transformer_layer_3/feedforward_output_dense/kernel_1:a�\
Z
_user_specified_nameB@transformer_layer_5/self_attention_layer/attention_output/bias_1:V�Q
O
_user_specified_name75transformer_layer_5/self_attention_layer_norm/gamma_1:V�Q
O
_user_specified_name75transformer_layer_0/self_attention_layer/value/bias_1:X�S
Q
_user_specified_name97transformer_layer_1/self_attention_layer/query/kernel_1:a�\
Z
_user_specified_nameB@transformer_layer_1/self_attention_layer/attention_output/bias_1:V�Q
O
_user_specified_name75transformer_layer_3/self_attention_layer_norm/gamma_1:Z�U
S
_user_specified_name;9transformer_layer_0/feedforward_intermediate_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_1/self_attention_layer/key/kernel_1:U�P
N
_user_specified_name64transformer_layer_1/self_attention_layer_norm/beta_1:V�Q
O
_user_specified_name75transformer_layer_4/feedforward_output_dense/kernel_1:X�S
Q
_user_specified_name97transformer_layer_4/self_attention_layer/query/kernel_1:V�Q
O
_user_specified_name75transformer_layer_0/feedforward_output_dense/kernel_1:S�N
L
_user_specified_name42transformer_layer_0/feedforward_layer_norm/gamma_1:T�O
M
_user_specified_name53transformer_layer_2/feedforward_output_dense/bias_1:V�Q
O
_user_specified_name75transformer_layer_5/feedforward_output_dense/kernel_1:>�9

_output_shapes
: 

_user_specified_nameConst"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serve�
D
padding_mask4
serve_padding_mask:0������������������
>
	token_ids1
serve_token_ids:0������������������<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict*�
serving_default�
N
padding_mask>
serving_default_padding_mask:0������������������
H
	token_ids;
serving_default_token_ids:0������������������>
output_02
StatefulPartitionedCall_1:0���������tensorflow/serving/predict:��
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures"
_generic_user_object
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30
'31
(32
)33
*34
+35
,36
-37
.38
/39
040
141
242
343
444
545
646
747
848
949
:50
;51
<52
=53
>54
?55
@56
A57
B58
C59
D60
E61
F62
G63
H64
I65
J66
K67
L68
M69
N70
O71
P72
Q73
R74
S75
T76
U77
V78
W79
X80
Y81
Z82
[83
\84
]85
^86
_87
`88
a89
b90
c91
d92
e93
f94
g95
h96
i97
j98
k99
l100
m101
n102
o103
p104
q105
r106
s107
t108
u109
v110
w111
x112
y113
z114
{115
|116
}117
~118
119
�120
�121
�122
�123"
trackable_list_wrapper
?
0
�1
�2
�3"
trackable_list_wrapper
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30
'31
(32
)33
*34
+35
,36
-37
.38
/39
040
141
242
343
444
545
646
747
848
949
:50
;51
<52
=53
>54
?55
@56
A57
B58
C59
D60
E61
F62
G63
H64
I65
J66
K67
L68
M69
N70
O71
P72
Q73
R74
S75
T76
U77
V78
W79
X80
Y81
Z82
[83
\84
]85
^86
_87
`88
a89
b90
c91
d92
e93
f94
g95
h96
i97
j98
k99
l100
m101
n102
o103
p104
q105
r106
s107
t108
u109
v110
w111
x112
y113
z114
{115
|116
}117
~118
�119"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67
�68
�69
�70
�71
�72
�73
�74
�75
�76
�77
�78
�79
�80
�81
�82
�83
�84
�85
�86
�87
�88
�89
�90
�91
�92
�93
�94
�95
�96
�97
�98
�99
�100
�101
�102
�103"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trace_02�
__inference___call___8096�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *g�d
b�_
/�,
padding_mask������������������
,�)
	token_ids������������������z�trace_0
9

�serve
�serving_default"
signature_map
L:J���27token_and_position_embedding/token_embedding/embeddings
N:L
��2:token_and_position_embedding/position_embedding/embeddings
*:(�2embeddings_layer_norm/gamma
):'�2embeddings_layer_norm/beta
 :2seed_generator_state
L:J�@25transformer_layer_0/self_attention_layer/query/kernel
E:C@23transformer_layer_0/self_attention_layer/query/bias
J:H�@23transformer_layer_0/self_attention_layer/key/kernel
C:A@21transformer_layer_0/self_attention_layer/key/bias
L:J�@25transformer_layer_0/self_attention_layer/value/kernel
E:C@23transformer_layer_0/self_attention_layer/value/bias
I:G2=transformer_layer_0/self_attention_layer/seed_generator_state
W:U@�2@transformer_layer_0/self_attention_layer/attention_output/kernel
M:K�2>transformer_layer_0/self_attention_layer/attention_output/bias
B:@�23transformer_layer_0/self_attention_layer_norm/gamma
A:?�22transformer_layer_0/self_attention_layer_norm/beta
4:22(transformer_layer_0/seed_generator_state
?:=�20transformer_layer_0/feedforward_layer_norm/gamma
>:<�2/transformer_layer_0/feedforward_layer_norm/beta
M:K
��29transformer_layer_0/feedforward_intermediate_dense/kernel
F:D�27transformer_layer_0/feedforward_intermediate_dense/bias
G:E
��23transformer_layer_0/feedforward_output_dense/kernel
@:>�21transformer_layer_0/feedforward_output_dense/bias
4:22(transformer_layer_0/seed_generator_state
L:J�@25transformer_layer_1/self_attention_layer/query/kernel
E:C@23transformer_layer_1/self_attention_layer/query/bias
J:H�@23transformer_layer_1/self_attention_layer/key/kernel
C:A@21transformer_layer_1/self_attention_layer/key/bias
L:J�@25transformer_layer_1/self_attention_layer/value/kernel
E:C@23transformer_layer_1/self_attention_layer/value/bias
I:G2=transformer_layer_1/self_attention_layer/seed_generator_state
W:U@�2@transformer_layer_1/self_attention_layer/attention_output/kernel
M:K�2>transformer_layer_1/self_attention_layer/attention_output/bias
B:@�23transformer_layer_1/self_attention_layer_norm/gamma
A:?�22transformer_layer_1/self_attention_layer_norm/beta
4:22(transformer_layer_1/seed_generator_state
?:=�20transformer_layer_1/feedforward_layer_norm/gamma
>:<�2/transformer_layer_1/feedforward_layer_norm/beta
M:K
��29transformer_layer_1/feedforward_intermediate_dense/kernel
F:D�27transformer_layer_1/feedforward_intermediate_dense/bias
G:E
��23transformer_layer_1/feedforward_output_dense/kernel
@:>�21transformer_layer_1/feedforward_output_dense/bias
4:22(transformer_layer_1/seed_generator_state
L:J�@25transformer_layer_2/self_attention_layer/query/kernel
E:C@23transformer_layer_2/self_attention_layer/query/bias
J:H�@23transformer_layer_2/self_attention_layer/key/kernel
C:A@21transformer_layer_2/self_attention_layer/key/bias
L:J�@25transformer_layer_2/self_attention_layer/value/kernel
E:C@23transformer_layer_2/self_attention_layer/value/bias
I:G2=transformer_layer_2/self_attention_layer/seed_generator_state
W:U@�2@transformer_layer_2/self_attention_layer/attention_output/kernel
M:K�2>transformer_layer_2/self_attention_layer/attention_output/bias
B:@�23transformer_layer_2/self_attention_layer_norm/gamma
A:?�22transformer_layer_2/self_attention_layer_norm/beta
4:22(transformer_layer_2/seed_generator_state
?:=�20transformer_layer_2/feedforward_layer_norm/gamma
>:<�2/transformer_layer_2/feedforward_layer_norm/beta
M:K
��29transformer_layer_2/feedforward_intermediate_dense/kernel
F:D�27transformer_layer_2/feedforward_intermediate_dense/bias
G:E
��23transformer_layer_2/feedforward_output_dense/kernel
@:>�21transformer_layer_2/feedforward_output_dense/bias
4:22(transformer_layer_2/seed_generator_state
L:J�@25transformer_layer_3/self_attention_layer/query/kernel
E:C@23transformer_layer_3/self_attention_layer/query/bias
J:H�@23transformer_layer_3/self_attention_layer/key/kernel
C:A@21transformer_layer_3/self_attention_layer/key/bias
L:J�@25transformer_layer_3/self_attention_layer/value/kernel
E:C@23transformer_layer_3/self_attention_layer/value/bias
I:G2=transformer_layer_3/self_attention_layer/seed_generator_state
W:U@�2@transformer_layer_3/self_attention_layer/attention_output/kernel
M:K�2>transformer_layer_3/self_attention_layer/attention_output/bias
B:@�23transformer_layer_3/self_attention_layer_norm/gamma
A:?�22transformer_layer_3/self_attention_layer_norm/beta
4:22(transformer_layer_3/seed_generator_state
?:=�20transformer_layer_3/feedforward_layer_norm/gamma
>:<�2/transformer_layer_3/feedforward_layer_norm/beta
M:K
��29transformer_layer_3/feedforward_intermediate_dense/kernel
F:D�27transformer_layer_3/feedforward_intermediate_dense/bias
G:E
��23transformer_layer_3/feedforward_output_dense/kernel
@:>�21transformer_layer_3/feedforward_output_dense/bias
4:22(transformer_layer_3/seed_generator_state
L:J�@25transformer_layer_4/self_attention_layer/query/kernel
E:C@23transformer_layer_4/self_attention_layer/query/bias
J:H�@23transformer_layer_4/self_attention_layer/key/kernel
C:A@21transformer_layer_4/self_attention_layer/key/bias
L:J�@25transformer_layer_4/self_attention_layer/value/kernel
E:C@23transformer_layer_4/self_attention_layer/value/bias
I:G2=transformer_layer_4/self_attention_layer/seed_generator_state
W:U@�2@transformer_layer_4/self_attention_layer/attention_output/kernel
M:K�2>transformer_layer_4/self_attention_layer/attention_output/bias
B:@�23transformer_layer_4/self_attention_layer_norm/gamma
A:?�22transformer_layer_4/self_attention_layer_norm/beta
4:22(transformer_layer_4/seed_generator_state
?:=�20transformer_layer_4/feedforward_layer_norm/gamma
>:<�2/transformer_layer_4/feedforward_layer_norm/beta
M:K
��29transformer_layer_4/feedforward_intermediate_dense/kernel
F:D�27transformer_layer_4/feedforward_intermediate_dense/bias
G:E
��23transformer_layer_4/feedforward_output_dense/kernel
@:>�21transformer_layer_4/feedforward_output_dense/bias
4:22(transformer_layer_4/seed_generator_state
L:J�@25transformer_layer_5/self_attention_layer/query/kernel
E:C@23transformer_layer_5/self_attention_layer/query/bias
J:H�@23transformer_layer_5/self_attention_layer/key/kernel
C:A@21transformer_layer_5/self_attention_layer/key/bias
L:J�@25transformer_layer_5/self_attention_layer/value/kernel
E:C@23transformer_layer_5/self_attention_layer/value/bias
I:G2=transformer_layer_5/self_attention_layer/seed_generator_state
W:U@�2@transformer_layer_5/self_attention_layer/attention_output/kernel
M:K�2>transformer_layer_5/self_attention_layer/attention_output/bias
B:@�23transformer_layer_5/self_attention_layer_norm/gamma
A:?�22transformer_layer_5/self_attention_layer_norm/beta
4:22(transformer_layer_5/seed_generator_state
?:=�20transformer_layer_5/feedforward_layer_norm/gamma
>:<�2/transformer_layer_5/feedforward_layer_norm/beta
M:K
��29transformer_layer_5/feedforward_intermediate_dense/kernel
F:D�27transformer_layer_5/feedforward_intermediate_dense/bias
G:E
��23transformer_layer_5/feedforward_output_dense/kernel
@:>�21transformer_layer_5/feedforward_output_dense/bias
4:22(transformer_layer_5/seed_generator_state
':%
��2pooled_dense/kernel
 :�2pooled_dense/bias
 :2seed_generator_state
 :	�2logits/kernel
:2logits/bias
L:J�@25transformer_layer_3/self_attention_layer/query/kernel
E:C@23transformer_layer_3/self_attention_layer/value/bias
M:K�2>transformer_layer_3/self_attention_layer/attention_output/bias
@:>�21transformer_layer_3/feedforward_output_dense/bias
A:?�22transformer_layer_5/self_attention_layer_norm/beta
E:C@23transformer_layer_1/self_attention_layer/query/bias
J:H�@23transformer_layer_3/self_attention_layer/key/kernel
A:?�22transformer_layer_3/self_attention_layer_norm/beta
C:A@21transformer_layer_1/self_attention_layer/key/bias
@:>�21transformer_layer_4/feedforward_output_dense/bias
?:=�20transformer_layer_4/feedforward_layer_norm/gamma
>:<�2/transformer_layer_0/feedforward_layer_norm/beta
E:C@23transformer_layer_4/self_attention_layer/query/bias
J:H�@23transformer_layer_4/self_attention_layer/key/kernel
L:J���27token_and_position_embedding/token_embedding/embeddings
@:>�21transformer_layer_5/feedforward_output_dense/bias
E:C@23transformer_layer_3/self_attention_layer/query/bias
C:A@21transformer_layer_3/self_attention_layer/key/bias
B:@�23transformer_layer_0/self_attention_layer_norm/gamma
>:<�2/transformer_layer_4/feedforward_layer_norm/beta
):'�2embeddings_layer_norm/beta
L:J�@25transformer_layer_2/self_attention_layer/query/kernel
?:=�20transformer_layer_2/feedforward_layer_norm/gamma
F:D�27transformer_layer_4/feedforward_intermediate_dense/bias
C:A@21transformer_layer_4/self_attention_layer/key/bias
L:J�@25transformer_layer_4/self_attention_layer/value/kernel
G:E
��23transformer_layer_1/feedforward_output_dense/kernel
W:U@�2@transformer_layer_0/self_attention_layer/attention_output/kernel
?:=�20transformer_layer_1/feedforward_layer_norm/gamma
F:D�27transformer_layer_1/feedforward_intermediate_dense/bias
 :�2pooled_dense/bias
E:C@23transformer_layer_2/self_attention_layer/query/bias
>:<�2/transformer_layer_2/feedforward_layer_norm/beta
M:K
��29transformer_layer_4/feedforward_intermediate_dense/kernel
W:U@�2@transformer_layer_4/self_attention_layer/attention_output/kernel
N:L
��2:token_and_position_embedding/position_embedding/embeddings
J:H�@23transformer_layer_2/self_attention_layer/key/kernel
F:D�27transformer_layer_2/feedforward_intermediate_dense/bias
E:C@23transformer_layer_4/self_attention_layer/value/bias
?:=�20transformer_layer_5/feedforward_layer_norm/gamma
 :	�2logits/kernel
*:(�2embeddings_layer_norm/gamma
L:J�@25transformer_layer_2/self_attention_layer/value/kernel
?:=�20transformer_layer_3/feedforward_layer_norm/gamma
B:@�23transformer_layer_4/self_attention_layer_norm/gamma
L:J�@25transformer_layer_5/self_attention_layer/query/kernel
L:J�@25transformer_layer_0/self_attention_layer/query/kernel
M:K�2>transformer_layer_0/self_attention_layer/attention_output/bias
>:<�2/transformer_layer_1/feedforward_layer_norm/beta
J:H�@23transformer_layer_5/self_attention_layer/key/kernel
J:H�@23transformer_layer_0/self_attention_layer/key/kernel
A:?�22transformer_layer_0/self_attention_layer_norm/beta
L:J�@25transformer_layer_5/self_attention_layer/value/kernel
L:J�@25transformer_layer_1/self_attention_layer/value/kernel
C:A@21transformer_layer_2/self_attention_layer/key/bias
M:K
��29transformer_layer_2/feedforward_intermediate_dense/kernel
M:K�2>transformer_layer_4/self_attention_layer/attention_output/bias
>:<�2/transformer_layer_5/feedforward_layer_norm/beta
':%
��2pooled_dense/kernel
:2logits/bias
@:>�21transformer_layer_0/feedforward_output_dense/bias
E:C@23transformer_layer_2/self_attention_layer/value/bias
W:U@�2@transformer_layer_2/self_attention_layer/attention_output/kernel
>:<�2/transformer_layer_3/feedforward_layer_norm/beta
A:?�22transformer_layer_4/self_attention_layer_norm/beta
E:C@23transformer_layer_5/self_attention_layer/query/bias
M:K
��29transformer_layer_5/feedforward_intermediate_dense/kernel
E:C@23transformer_layer_0/self_attention_layer/query/bias
B:@�23transformer_layer_2/self_attention_layer_norm/gamma
M:K
��29transformer_layer_3/feedforward_intermediate_dense/kernel
C:A@21transformer_layer_0/self_attention_layer/key/bias
M:K
��29transformer_layer_1/feedforward_intermediate_dense/kernel
E:C@23transformer_layer_5/self_attention_layer/value/bias
W:U@�2@transformer_layer_5/self_attention_layer/attention_output/kernel
L:J�@25transformer_layer_0/self_attention_layer/value/kernel
E:C@23transformer_layer_1/self_attention_layer/value/bias
W:U@�2@transformer_layer_1/self_attention_layer/attention_output/kernel
M:K
��29transformer_layer_0/feedforward_intermediate_dense/kernel
B:@�23transformer_layer_1/self_attention_layer_norm/gamma
M:K�2>transformer_layer_2/self_attention_layer/attention_output/bias
G:E
��23transformer_layer_2/feedforward_output_dense/kernel
F:D�27transformer_layer_5/feedforward_intermediate_dense/bias
A:?�22transformer_layer_2/self_attention_layer_norm/beta
F:D�27transformer_layer_3/feedforward_intermediate_dense/bias
C:A@21transformer_layer_5/self_attention_layer/key/bias
@:>�21transformer_layer_1/feedforward_output_dense/bias
L:J�@25transformer_layer_3/self_attention_layer/value/kernel
W:U@�2@transformer_layer_3/self_attention_layer/attention_output/kernel
G:E
��23transformer_layer_3/feedforward_output_dense/kernel
M:K�2>transformer_layer_5/self_attention_layer/attention_output/bias
B:@�23transformer_layer_5/self_attention_layer_norm/gamma
E:C@23transformer_layer_0/self_attention_layer/value/bias
L:J�@25transformer_layer_1/self_attention_layer/query/kernel
M:K�2>transformer_layer_1/self_attention_layer/attention_output/bias
B:@�23transformer_layer_3/self_attention_layer_norm/gamma
F:D�27transformer_layer_0/feedforward_intermediate_dense/bias
J:H�@23transformer_layer_1/self_attention_layer/key/kernel
A:?�22transformer_layer_1/self_attention_layer_norm/beta
G:E
��23transformer_layer_4/feedforward_output_dense/kernel
L:J�@25transformer_layer_4/self_attention_layer/query/kernel
G:E
��23transformer_layer_0/feedforward_output_dense/kernel
?:=�20transformer_layer_0/feedforward_layer_norm/gamma
@:>�21transformer_layer_2/feedforward_output_dense/bias
G:E
��23transformer_layer_5/feedforward_output_dense/kernel
�B�
__inference___call___8096padding_mask	token_ids"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_signature_wrapper___call___8311padding_mask	token_ids"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 .

kwonlyargs �
jpadding_mask
j	token_ids
kwonlydefaults
 
annotations� *
 
�B�
+__inference_signature_wrapper___call___8525padding_mask	token_ids"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 .

kwonlyargs �
jpadding_mask
j	token_ids
kwonlydefaults
 
annotations� *
 �
__inference___call___8096�k	
 !"#$%'()*./01,-345678:;<=ABCD?@FGHIJKMNOPTUVWRSYZ[\]^`abcghijeflmnopqstuvz{|}xy���q�n
g�d
b�_
/�,
padding_mask������������������
,�)
	token_ids������������������
� "!�
unknown����������
+__inference_signature_wrapper___call___8311�k	
 !"#$%'()*./01,-345678:;<=ABCD?@FGHIJKMNOPTUVWRSYZ[\]^`abcghijeflmnopqstuvz{|}xy������
� 
�|
?
padding_mask/�,
padding_mask������������������
9
	token_ids,�)
	token_ids������������������"3�0
.
output_0"�
output_0����������
+__inference_signature_wrapper___call___8525�k	
 !"#$%'()*./01,-345678:;<=ABCD?@FGHIJKMNOPTUVWRSYZ[\]^`abcghijeflmnopqstuvz{|}xy������
� 
�|
?
padding_mask/�,
padding_mask������������������
9
	token_ids,�)
	token_ids������������������"3�0
.
output_0"�
output_0���������