╥Ю
╦Ь
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
6
Pow
x"T
y"T
z"T"
Ttype:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8Юс
И
convLayer1_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconvLayer1_weights
Б
&convLayer1_weights/Read/ReadVariableOpReadVariableOpconvLayer1_weights*&
_output_shapes
: *
dtype0
v
convLayer1_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconvLayer1_bias
o
#convLayer1_bias/Read/ReadVariableOpReadVariableOpconvLayer1_bias*
_output_shapes
: *
dtype0
И
convLayer2_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_nameconvLayer2_weights
Б
&convLayer2_weights/Read/ReadVariableOpReadVariableOpconvLayer2_weights*&
_output_shapes
:  *
dtype0
v
convLayer2_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconvLayer2_bias
o
#convLayer2_bias/Read/ReadVariableOpReadVariableOpconvLayer2_bias*
_output_shapes
: *
dtype0
И
convLayer3_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_nameconvLayer3_weights
Б
&convLayer3_weights/Read/ReadVariableOpReadVariableOpconvLayer3_weights*&
_output_shapes
:  *
dtype0
v
convLayer3_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconvLayer3_bias
o
#convLayer3_bias/Read/ReadVariableOpReadVariableOpconvLayer3_bias*
_output_shapes
: *
dtype0
И
convLayer4_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_nameconvLayer4_weights
Б
&convLayer4_weights/Read/ReadVariableOpReadVariableOpconvLayer4_weights*&
_output_shapes
:  *
dtype0
v
convLayer4_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconvLayer4_bias
o
#convLayer4_bias/Read/ReadVariableOpReadVariableOpconvLayer4_bias*
_output_shapes
: *
dtype0
И
convLayer5_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *#
shared_nameconvLayer5_weights
Б
&convLayer5_weights/Read/ReadVariableOpReadVariableOpconvLayer5_weights*&
_output_shapes
:@ *
dtype0
v
convLayer5_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconvLayer5_bias
o
#convLayer5_bias/Read/ReadVariableOpReadVariableOpconvLayer5_bias*
_output_shapes
: *
dtype0
И
convLayer6_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *#
shared_nameconvLayer6_weights
Б
&convLayer6_weights/Read/ReadVariableOpReadVariableOpconvLayer6_weights*&
_output_shapes
:@ *
dtype0
v
convLayer6_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconvLayer6_bias
o
#convLayer6_bias/Read/ReadVariableOpReadVariableOpconvLayer6_bias*
_output_shapes
: *
dtype0
И
convLayer7_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameconvLayer7_weights
Б
&convLayer7_weights/Read/ReadVariableOpReadVariableOpconvLayer7_weights*&
_output_shapes
:@*
dtype0
v
convLayer7_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconvLayer7_bias
o
#convLayer7_bias/Read/ReadVariableOpReadVariableOpconvLayer7_bias*
_output_shapes
:*
dtype0

NoOpNoOp
Ф
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╧
value┼B┬ B╗
!
forwardPass

signatures
1
0
1
2
3
4
5
	6
 



kernel
bias


kernel
bias


kernel
bias


kernel
bias


kernel
bias


kernel
bias


kernel
bias
WU
VARIABLE_VALUEconvLayer1_weights/forwardPass/0/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconvLayer1_bias-forwardPass/0/bias/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconvLayer2_weights/forwardPass/1/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconvLayer2_bias-forwardPass/1/bias/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconvLayer3_weights/forwardPass/2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconvLayer3_bias-forwardPass/2/bias/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconvLayer4_weights/forwardPass/3/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconvLayer4_bias-forwardPass/3/bias/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconvLayer5_weights/forwardPass/4/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconvLayer5_bias-forwardPass/4/bias/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconvLayer6_weights/forwardPass/5/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconvLayer6_bias-forwardPass/5/bias/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconvLayer7_weights/forwardPass/6/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconvLayer7_bias-forwardPass/6/bias/.ATTRIBUTES/VARIABLE_VALUE
K
serving_default_inputPlaceholder*
_output_shapes
:*
dtype0
Ё
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconvLayer1_weightsconvLayer1_biasconvLayer2_weightsconvLayer2_biasconvLayer3_weightsconvLayer3_biasconvLayer4_weightsconvLayer4_biasconvLayer5_weightsconvLayer5_biasconvLayer6_weightsconvLayer6_biasconvLayer7_weightsconvLayer7_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:::+                           *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference_signature_wrapper_441
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┴
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&convLayer1_weights/Read/ReadVariableOp#convLayer1_bias/Read/ReadVariableOp&convLayer2_weights/Read/ReadVariableOp#convLayer2_bias/Read/ReadVariableOp&convLayer3_weights/Read/ReadVariableOp#convLayer3_bias/Read/ReadVariableOp&convLayer4_weights/Read/ReadVariableOp#convLayer4_bias/Read/ReadVariableOp&convLayer5_weights/Read/ReadVariableOp#convLayer5_bias/Read/ReadVariableOp&convLayer6_weights/Read/ReadVariableOp#convLayer6_bias/Read/ReadVariableOp&convLayer7_weights/Read/ReadVariableOp#convLayer7_bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8В *%
f R
__inference__traced_save_785
д
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconvLayer1_weightsconvLayer1_biasconvLayer2_weightsconvLayer2_biasconvLayer3_weightsconvLayer3_biasconvLayer4_weightsconvLayer4_biasconvLayer5_weightsconvLayer5_biasconvLayer6_weightsconvLayer6_biasconvLayer7_weightsconvLayer7_bias*
Tin
2*
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
GPU 2J 8В *(
f#R!
__inference__traced_restore_837ки
И
╞
__inference___call___682
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                            
%
_user_specified_nameinputTensor
И
╞
__inference___call___474
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                            
%
_user_specified_nameinputTensor
И
╞
__inference___call___237
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                            
%
_user_specified_nameinputTensor
И
╞
__inference___call___706
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                           @
%
_user_specified_nameinputTensor
Ї
╞
__inference___call___570
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Tanhз
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identityл

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                           @
%
_user_specified_nameinputTensor
Ї
╞
__inference___call___718
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Tanhз
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identityл

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                           @
%
_user_specified_nameinputTensor
И
╞
__inference___call___510
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                            
%
_user_specified_nameinputTensor
н(
Ю
__inference__traced_save_785
file_prefix1
-savev2_convlayer1_weights_read_readvariableop.
*savev2_convlayer1_bias_read_readvariableop1
-savev2_convlayer2_weights_read_readvariableop.
*savev2_convlayer2_bias_read_readvariableop1
-savev2_convlayer3_weights_read_readvariableop.
*savev2_convlayer3_bias_read_readvariableop1
-savev2_convlayer4_weights_read_readvariableop.
*savev2_convlayer4_bias_read_readvariableop1
-savev2_convlayer5_weights_read_readvariableop.
*savev2_convlayer5_bias_read_readvariableop1
-savev2_convlayer6_weights_read_readvariableop.
*savev2_convlayer6_bias_read_readvariableop1
-savev2_convlayer7_weights_read_readvariableop.
*savev2_convlayer7_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename┴
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╙
value╔B╞B/forwardPass/0/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/0/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/1/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/1/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/2/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/2/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/3/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/3/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/4/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/4/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/5/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/5/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/6/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesж
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices┼
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_convlayer1_weights_read_readvariableop*savev2_convlayer1_bias_read_readvariableop-savev2_convlayer2_weights_read_readvariableop*savev2_convlayer2_bias_read_readvariableop-savev2_convlayer3_weights_read_readvariableop*savev2_convlayer3_bias_read_readvariableop-savev2_convlayer4_weights_read_readvariableop*savev2_convlayer4_bias_read_readvariableop-savev2_convlayer5_weights_read_readvariableop*savev2_convlayer5_bias_read_readvariableop-savev2_convlayer6_weights_read_readvariableop*savev2_convlayer6_bias_read_readvariableop-savev2_convlayer7_weights_read_readvariableop*savev2_convlayer7_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*┴
_input_shapesп
м: : : :  : :  : :  : :@ : :@ : :@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
:@ : 


_output_shapes
: :,(
&
_output_shapes
:@ : 

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
::

_output_shapes
: 
И
╞
__inference___call___317
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                           @
%
_user_specified_nameinputTensor
╢
╞
__inference___call___646
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:E A

_output_shapes
:
%
_user_specified_nameinputTensor
ЕS
э
__inference___call___634	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

identity_2

identity_3

identity_4ИвStatefulPartitionedCallвStatefulPartitionedCall_1вStatefulPartitionedCall_2вStatefulPartitionedCall_3вStatefulPartitionedCall_4вStatefulPartitionedCall_5вStatefulPartitionedCall_6J
IdentityIdentityinput*
T0*
_output_shapes
:2

IdentityП
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___4562
StatefulPartitionedCall░
StatefulPartitionedCall_1StatefulPartitionedCall StatefulPartitionedCall:output:1	unknown_1	unknown_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___4742
StatefulPartitionedCall_1▓
StatefulPartitionedCall_2StatefulPartitionedCall"StatefulPartitionedCall_1:output:1	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___4922
StatefulPartitionedCall_2▓
StatefulPartitionedCall_3StatefulPartitionedCall"StatefulPartitionedCall_2:output:1	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___5102
StatefulPartitionedCall_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╧
concatConcatV2"StatefulPartitionedCall_3:output:1"StatefulPartitionedCall_2:output:0concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           @2
concatЯ
StatefulPartitionedCall_4StatefulPartitionedCallconcat:output:0	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___5302
StatefulPartitionedCall_4`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis╒
concat_1ConcatV2"StatefulPartitionedCall_4:output:1"StatefulPartitionedCall_1:output:0concat_1/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           @2

concat_1в
StatefulPartitionedCall_5StatefulPartitionedCallconcat_1:output:0	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___5502
StatefulPartitionedCall_5`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis╙
concat_2ConcatV2"StatefulPartitionedCall_5:output:1 StatefulPartitionedCall:output:0concat_2/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           @2

concat_2г
StatefulPartitionedCall_6StatefulPartitionedCallconcat_2:output:0
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                           :+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___5702
StatefulPartitionedCall_6Д
TanhTanh"StatefulPartitionedCall_6:output:1*
T0*A
_output_shapes/
-:+                           2
TanhP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim╓
splitSplitsplit/split_dim:output:0Tanh:y:0*
T0*■
_output_shapesы
ш:+                           :+                           :+                           :+                           :+                           :+                           :+                           :+                           *
	num_split2
splitS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Pow/yW
PowPowIdentity:output:0Pow/y:output:0*
T0*
_output_shapes
:2
PowP
subSubPow:z:0Identity:output:0*
T0*
_output_shapes
:2
subM
mulMulsplit:output:0sub:z:0*
T0*
_output_shapes
:2
mulR
addAddV2Identity:output:0mul:z:0*
T0*
_output_shapes
:2
addW
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_1/yS
Pow_1Powadd:z:0Pow_1/y:output:0*
T0*
_output_shapes
:2
Pow_1L
sub_1Sub	Pow_1:z:0add:z:0*
T0*
_output_shapes
:2
sub_1S
mul_1Mulsplit:output:1	sub_1:z:0*
T0*
_output_shapes
:2
mul_1N
add_1AddV2add:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1W
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_2/yU
Pow_2Pow	add_1:z:0Pow_2/y:output:0*
T0*
_output_shapes
:2
Pow_2N
sub_2Sub	Pow_2:z:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2S
mul_2Mulsplit:output:2	sub_2:z:0*
T0*
_output_shapes
:2
mul_2P
add_2AddV2	add_1:z:0	mul_2:z:0*
T0*
_output_shapes
:2
add_2W
Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_3/yU
Pow_3Pow	add_2:z:0Pow_3/y:output:0*
T0*
_output_shapes
:2
Pow_3N
sub_3Sub	Pow_3:z:0	add_2:z:0*
T0*
_output_shapes
:2
sub_3S
mul_3Mulsplit:output:3	sub_3:z:0*
T0*
_output_shapes
:2
mul_3P
add_3AddV2	add_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_3W
Pow_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_4/yU
Pow_4Pow	add_3:z:0Pow_4/y:output:0*
T0*
_output_shapes
:2
Pow_4N
sub_4Sub	Pow_4:z:0	add_3:z:0*
T0*
_output_shapes
:2
sub_4S
mul_4Mulsplit:output:4	sub_4:z:0*
T0*
_output_shapes
:2
mul_4P
add_4AddV2	add_3:z:0	mul_4:z:0*
T0*
_output_shapes
:2
add_4R

Identity_1Identity	add_4:z:0*
T0*
_output_shapes
:2

Identity_1W
Pow_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_5/yU
Pow_5Pow	add_4:z:0Pow_5/y:output:0*
T0*
_output_shapes
:2
Pow_5N
sub_5Sub	Pow_5:z:0	add_4:z:0*
T0*
_output_shapes
:2
sub_5S
mul_5Mulsplit:output:5	sub_5:z:0*
T0*
_output_shapes
:2
mul_5P
add_5AddV2	add_4:z:0	mul_5:z:0*
T0*
_output_shapes
:2
add_5W
Pow_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_6/yU
Pow_6Pow	add_5:z:0Pow_6/y:output:0*
T0*
_output_shapes
:2
Pow_6N
sub_6Sub	Pow_6:z:0	add_5:z:0*
T0*
_output_shapes
:2
sub_6S
mul_6Mulsplit:output:6	sub_6:z:0*
T0*
_output_shapes
:2
mul_6P
add_6AddV2	add_5:z:0	mul_6:z:0*
T0*
_output_shapes
:2
add_6W
Pow_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_7/yU
Pow_7Pow	add_6:z:0Pow_7/y:output:0*
T0*
_output_shapes
:2
Pow_7N
sub_7Sub	Pow_7:z:0	add_6:z:0*
T0*
_output_shapes
:2
sub_7S
mul_7Mulsplit:output:7	sub_7:z:0*
T0*
_output_shapes
:2
mul_7P
add_7AddV2	add_6:z:0	mul_7:z:0*
T0*
_output_shapes
:2
add_7`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_3/axisН
concat_3ConcatV2split:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7concat_3/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           2

concat_3Ю

Identity_2IdentityIdentity_1:output:0^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6*
T0*
_output_shapes
:2

Identity_2Ф

Identity_3Identity	add_7:z:0^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6*
T0*
_output_shapes
:2

Identity_3┼

Identity_4Identityconcat_3:output:0^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6*
T0*A
_output_shapes/
-:+                           2

Identity_4"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*O
_input_shapes>
<:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall26
StatefulPartitionedCall_1StatefulPartitionedCall_126
StatefulPartitionedCall_2StatefulPartitionedCall_226
StatefulPartitionedCall_3StatefulPartitionedCall_326
StatefulPartitionedCall_4StatefulPartitionedCall_426
StatefulPartitionedCall_5StatefulPartitionedCall_526
StatefulPartitionedCall_6StatefulPartitionedCall_6:? ;

_output_shapes
:

_user_specified_nameinput
Є<
╡
__inference__traced_restore_837
file_prefix'
#assignvariableop_convlayer1_weights&
"assignvariableop_1_convlayer1_bias)
%assignvariableop_2_convlayer2_weights&
"assignvariableop_3_convlayer2_bias)
%assignvariableop_4_convlayer3_weights&
"assignvariableop_5_convlayer3_bias)
%assignvariableop_6_convlayer4_weights&
"assignvariableop_7_convlayer4_bias)
%assignvariableop_8_convlayer5_weights&
"assignvariableop_9_convlayer5_bias*
&assignvariableop_10_convlayer6_weights'
#assignvariableop_11_convlayer6_bias*
&assignvariableop_12_convlayer7_weights'
#assignvariableop_13_convlayer7_bias
identity_15ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╟
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╙
value╔B╞B/forwardPass/0/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/0/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/1/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/1/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/2/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/2/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/3/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/3/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/4/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/4/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/5/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/5/bias/.ATTRIBUTES/VARIABLE_VALUEB/forwardPass/6/kernel/.ATTRIBUTES/VARIABLE_VALUEB-forwardPass/6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesм
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityв
AssignVariableOpAssignVariableOp#assignvariableop_convlayer1_weightsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1з
AssignVariableOp_1AssignVariableOp"assignvariableop_1_convlayer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2к
AssignVariableOp_2AssignVariableOp%assignvariableop_2_convlayer2_weightsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3з
AssignVariableOp_3AssignVariableOp"assignvariableop_3_convlayer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4к
AssignVariableOp_4AssignVariableOp%assignvariableop_4_convlayer3_weightsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5з
AssignVariableOp_5AssignVariableOp"assignvariableop_5_convlayer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6к
AssignVariableOp_6AssignVariableOp%assignvariableop_6_convlayer4_weightsIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7з
AssignVariableOp_7AssignVariableOp"assignvariableop_7_convlayer4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8к
AssignVariableOp_8AssignVariableOp%assignvariableop_8_convlayer5_weightsIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9з
AssignVariableOp_9AssignVariableOp"assignvariableop_9_convlayer5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10о
AssignVariableOp_10AssignVariableOp&assignvariableop_10_convlayer6_weightsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11л
AssignVariableOp_11AssignVariableOp#assignvariableop_11_convlayer6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12о
AssignVariableOp_12AssignVariableOp&assignvariableop_12_convlayer7_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13л
AssignVariableOp_13AssignVariableOp#assignvariableop_13_convlayer7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpТ
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14Е
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ї
╞
__inference___call___338
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Tanhз
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identityл

Identity_1IdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                           @
%
_user_specified_nameinputTensor
╢
╞
__inference___call___218
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:E A

_output_shapes
:
%
_user_specified_nameinputTensor
╢
╞
__inference___call___456
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:E A

_output_shapes
:
%
_user_specified_nameinputTensor
И
╞
__inference___call___275
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                            
%
_user_specified_nameinputTensor
И
╞
__inference___call___658
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                            
%
_user_specified_nameinputTensor
И
╞
__inference___call___530
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                           @
%
_user_specified_nameinputTensor
И
╞
__inference___call___550
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                           @
%
_user_specified_nameinputTensor
ЕS
э
__inference___call___402	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

identity_2

identity_3

identity_4ИвStatefulPartitionedCallвStatefulPartitionedCall_1вStatefulPartitionedCall_2вStatefulPartitionedCall_3вStatefulPartitionedCall_4вStatefulPartitionedCall_5вStatefulPartitionedCall_6J
IdentityIdentityinput*
T0*
_output_shapes
:2

IdentityП
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___2182
StatefulPartitionedCall░
StatefulPartitionedCall_1StatefulPartitionedCall StatefulPartitionedCall:output:1	unknown_1	unknown_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___2372
StatefulPartitionedCall_1▓
StatefulPartitionedCall_2StatefulPartitionedCall"StatefulPartitionedCall_1:output:1	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___2562
StatefulPartitionedCall_2▓
StatefulPartitionedCall_3StatefulPartitionedCall"StatefulPartitionedCall_2:output:1	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___2752
StatefulPartitionedCall_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis╧
concatConcatV2"StatefulPartitionedCall_3:output:1"StatefulPartitionedCall_2:output:0concat/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           @2
concatЯ
StatefulPartitionedCall_4StatefulPartitionedCallconcat:output:0	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___2962
StatefulPartitionedCall_4`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis╒
concat_1ConcatV2"StatefulPartitionedCall_4:output:1"StatefulPartitionedCall_1:output:0concat_1/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           @2

concat_1в
StatefulPartitionedCall_5StatefulPartitionedCallconcat_1:output:0	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                            :+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___3172
StatefulPartitionedCall_5`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis╙
concat_2ConcatV2"StatefulPartitionedCall_5:output:1 StatefulPartitionedCall:output:0concat_2/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           @2

concat_2г
StatefulPartitionedCall_6StatefulPartitionedCallconcat_2:output:0
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *n
_output_shapes\
Z:+                           :+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___3382
StatefulPartitionedCall_6Д
TanhTanh"StatefulPartitionedCall_6:output:1*
T0*A
_output_shapes/
-:+                           2
TanhP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim╓
splitSplitsplit/split_dim:output:0Tanh:y:0*
T0*■
_output_shapesы
ш:+                           :+                           :+                           :+                           :+                           :+                           :+                           :+                           *
	num_split2
splitS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Pow/yW
PowPowIdentity:output:0Pow/y:output:0*
T0*
_output_shapes
:2
PowP
subSubPow:z:0Identity:output:0*
T0*
_output_shapes
:2
subM
mulMulsplit:output:0sub:z:0*
T0*
_output_shapes
:2
mulR
addAddV2Identity:output:0mul:z:0*
T0*
_output_shapes
:2
addW
Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_1/yS
Pow_1Powadd:z:0Pow_1/y:output:0*
T0*
_output_shapes
:2
Pow_1L
sub_1Sub	Pow_1:z:0add:z:0*
T0*
_output_shapes
:2
sub_1S
mul_1Mulsplit:output:1	sub_1:z:0*
T0*
_output_shapes
:2
mul_1N
add_1AddV2add:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1W
Pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_2/yU
Pow_2Pow	add_1:z:0Pow_2/y:output:0*
T0*
_output_shapes
:2
Pow_2N
sub_2Sub	Pow_2:z:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2S
mul_2Mulsplit:output:2	sub_2:z:0*
T0*
_output_shapes
:2
mul_2P
add_2AddV2	add_1:z:0	mul_2:z:0*
T0*
_output_shapes
:2
add_2W
Pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_3/yU
Pow_3Pow	add_2:z:0Pow_3/y:output:0*
T0*
_output_shapes
:2
Pow_3N
sub_3Sub	Pow_3:z:0	add_2:z:0*
T0*
_output_shapes
:2
sub_3S
mul_3Mulsplit:output:3	sub_3:z:0*
T0*
_output_shapes
:2
mul_3P
add_3AddV2	add_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_3W
Pow_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_4/yU
Pow_4Pow	add_3:z:0Pow_4/y:output:0*
T0*
_output_shapes
:2
Pow_4N
sub_4Sub	Pow_4:z:0	add_3:z:0*
T0*
_output_shapes
:2
sub_4S
mul_4Mulsplit:output:4	sub_4:z:0*
T0*
_output_shapes
:2
mul_4P
add_4AddV2	add_3:z:0	mul_4:z:0*
T0*
_output_shapes
:2
add_4R

Identity_1Identity	add_4:z:0*
T0*
_output_shapes
:2

Identity_1W
Pow_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_5/yU
Pow_5Pow	add_4:z:0Pow_5/y:output:0*
T0*
_output_shapes
:2
Pow_5N
sub_5Sub	Pow_5:z:0	add_4:z:0*
T0*
_output_shapes
:2
sub_5S
mul_5Mulsplit:output:5	sub_5:z:0*
T0*
_output_shapes
:2
mul_5P
add_5AddV2	add_4:z:0	mul_5:z:0*
T0*
_output_shapes
:2
add_5W
Pow_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_6/yU
Pow_6Pow	add_5:z:0Pow_6/y:output:0*
T0*
_output_shapes
:2
Pow_6N
sub_6Sub	Pow_6:z:0	add_5:z:0*
T0*
_output_shapes
:2
sub_6S
mul_6Mulsplit:output:6	sub_6:z:0*
T0*
_output_shapes
:2
mul_6P
add_6AddV2	add_5:z:0	mul_6:z:0*
T0*
_output_shapes
:2
add_6W
Pow_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
Pow_7/yU
Pow_7Pow	add_6:z:0Pow_7/y:output:0*
T0*
_output_shapes
:2
Pow_7N
sub_7Sub	Pow_7:z:0	add_6:z:0*
T0*
_output_shapes
:2
sub_7S
mul_7Mulsplit:output:7	sub_7:z:0*
T0*
_output_shapes
:2
mul_7P
add_7AddV2	add_6:z:0	mul_7:z:0*
T0*
_output_shapes
:2
add_7`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_3/axisН
concat_3ConcatV2split:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7concat_3/axis:output:0*
N*
T0*A
_output_shapes/
-:+                           2

concat_3Ю

Identity_2IdentityIdentity_1:output:0^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6*
T0*
_output_shapes
:2

Identity_2Ф

Identity_3Identity	add_7:z:0^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6*
T0*
_output_shapes
:2

Identity_3┼

Identity_4Identityconcat_3:output:0^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6*
T0*A
_output_shapes/
-:+                           2

Identity_4"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*O
_input_shapes>
<:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall26
StatefulPartitionedCall_1StatefulPartitionedCall_126
StatefulPartitionedCall_2StatefulPartitionedCall_226
StatefulPartitionedCall_3StatefulPartitionedCall_326
StatefulPartitionedCall_4StatefulPartitionedCall_426
StatefulPartitionedCall_5StatefulPartitionedCall_526
StatefulPartitionedCall_6StatefulPartitionedCall_6:? ;

_output_shapes
:

_user_specified_nameinput
И
╞
__inference___call___492
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                            
%
_user_specified_nameinputTensor
И
╞
__inference___call___670
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                            
%
_user_specified_nameinputTensor
╜
╠
!__inference_signature_wrapper_441	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity

identity_1

identity_2ИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:::+                           *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference___call___4022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

IdentityГ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1м

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*O
_input_shapes>
<:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:? ;

_output_shapes
:

_user_specified_nameinput
И
╞
__inference___call___296
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                           @
%
_user_specified_nameinputTensor
И
╞
__inference___call___256
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                            
%
_user_specified_nameinputTensor
И
╞
__inference___call___694
inputtensor"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1ИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp║
Conv2DConv2DinputtensorConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity╡

Identity_1IdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:n j
A
_output_shapes/
-:+                           @
%
_user_specified_nameinputTensor"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Р
serving_default№
(
input
serving_default_input:0-
output_0!
StatefulPartitionedCall:0-
output_1!
StatefulPartitionedCall:1V
output_2J
StatefulPartitionedCall:2+                           tensorflow/serving/predict:Э'
M
forwardPass

signatures
__call__"
_generic_user_object
Q
0
1
2
3
4
5
	6"
trackable_list_wrapper
,
serving_default"
signature_map
B


kernel
bias
__call__"
_generic_user_object
B

kernel
bias
__call__"
_generic_user_object
B

kernel
bias
__call__"
_generic_user_object
B

kernel
bias
__call__"
_generic_user_object
B

kernel
bias
__call__"
_generic_user_object
B

kernel
bias
__call__"
_generic_user_object
B

kernel
bias
 __call__"
_generic_user_object
,:* 2convLayer1_weights
: 2convLayer1_bias
,:*  2convLayer2_weights
: 2convLayer2_bias
,:*  2convLayer3_weights
: 2convLayer3_bias
,:*  2convLayer4_weights
: 2convLayer4_bias
,:*@ 2convLayer5_weights
: 2convLayer5_bias
,:*@ 2convLayer6_weights
: 2convLayer6_bias
,:*@2convLayer7_weights
:2convLayer7_bias
╜2║
__inference___call___634Э
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╞B├
!__inference_signature_wrapper_441input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╟2─
__inference___call___646з
Ю▓Ъ
FullArgSpec"
argsЪ
jself
jinputTensor
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╟2─
__inference___call___658з
Ю▓Ъ
FullArgSpec"
argsЪ
jself
jinputTensor
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╟2─
__inference___call___670з
Ю▓Ъ
FullArgSpec"
argsЪ
jself
jinputTensor
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╟2─
__inference___call___682з
Ю▓Ъ
FullArgSpec"
argsЪ
jself
jinputTensor
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╟2─
__inference___call___694з
Ю▓Ъ
FullArgSpec"
argsЪ
jself
jinputTensor
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╟2─
__inference___call___706з
Ю▓Ъ
FullArgSpec"
argsЪ
jself
jinputTensor
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╟2─
__inference___call___718з
Ю▓Ъ
FullArgSpec"
argsЪ
jself
jinputTensor
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ж
__inference___call___634Й
в
в
К
input
к "VвS
К	
0
К	
1
5К2
2+                           ╗
__inference___call___646Ю
%в"
в
К
inputTensor
к "qвn
5К2
0+                            
5К2
1+                            ф
__inference___call___658╟NвK
DвA
?К<
inputTensor+                            
к "qвn
5К2
0+                            
5К2
1+                            ф
__inference___call___670╟NвK
DвA
?К<
inputTensor+                            
к "qвn
5К2
0+                            
5К2
1+                            ф
__inference___call___682╟NвK
DвA
?К<
inputTensor+                            
к "qвn
5К2
0+                            
5К2
1+                            ф
__inference___call___694╟NвK
DвA
?К<
inputTensor+                           @
к "qвn
5К2
0+                            
5К2
1+                            ф
__inference___call___706╟NвK
DвA
?К<
inputTensor+                           @
к "qвn
5К2
0+                            
5К2
1+                            ф
__inference___call___718╟NвK
DвA
?К<
inputTensor+                           @
к "qвn
5К2
0+                           
5К2
1+                           є
!__inference_signature_wrapper_441═
(в%
в 
к

inputК
input"РкМ

output_0К
output_0

output_1К
output_1
H
output_2<К9
output_2+                           