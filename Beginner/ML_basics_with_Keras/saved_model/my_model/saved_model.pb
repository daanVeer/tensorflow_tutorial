Вс
ЛБ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
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
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.3.02v2.3.0-rc2-23-gb36436b0878эП
z
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
љђ*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
љђ*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:ђ*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ
*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	ђ
*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
ѕ
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
љђ*&
shared_nameAdam/dense_8/kernel/m
Ђ
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m* 
_output_shapes
:
љђ*
dtype0

Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*$
shared_nameAdam/dense_8/bias/m
x
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes	
:ђ*
dtype0
Є
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ
*&
shared_nameAdam/dense_9/kernel/m
ђ
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes
:	ђ
*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:
*
dtype0
ѕ
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
љђ*&
shared_nameAdam/dense_8/kernel/v
Ђ
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v* 
_output_shapes
:
љђ*
dtype0

Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*$
shared_nameAdam/dense_8/bias/v
x
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes	
:ђ*
dtype0
Є
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ
*&
shared_nameAdam/dense_9/kernel/v
ђ
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes
:	ђ
*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
о
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Љ
valueЄBё B§
╠
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
h


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
ѕ
iter

beta_1

beta_2
	decay
learning_rate
m>m?m@mA
vBvCvDvE


0
1
2
3


0
1
2
3
 
Г
trainable_variables
	variables
layer_metrics

 layers
!non_trainable_variables
regularization_losses
"metrics
#layer_regularization_losses
 
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1


0
1
 
Г
trainable_variables
	variables
$layer_metrics

%layers
&non_trainable_variables
regularization_losses
'metrics
(layer_regularization_losses
 
 
 
Г
trainable_variables
	variables
)layer_metrics

*layers
+non_trainable_variables
regularization_losses
,metrics
-layer_regularization_losses
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Г
trainable_variables
	variables
.layer_metrics

/layers
0non_trainable_variables
regularization_losses
1metrics
2layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
 

30
41
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	5total
	6count
7	variables
8	keras_api
D
	9total
	:count
;
_fn_kwargs
<	variables
=	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

50
61

7	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

<	variables
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѓ
serving_default_dense_8_inputPlaceholder*(
_output_shapes
:         љ*
dtype0*
shape:         љ
ч
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_8_inputdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference_signature_wrapper_13301
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ц
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
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
GPU 2J 8ѓ *'
f"R 
__inference__traced_save_13522
ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasdense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v*!
Tin
2*
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
GPU 2J 8ѓ **
f%R#
!__inference__traced_restore_13595иЄ
Ш
ў
G__inference_sequential_4_layer_call_and_return_conditional_losses_13326

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityѕД
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
љђ*
dtype02
dense_8/MatMul/ReadVariableOpї
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_8/MatMulЦ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_8/BiasAdd/ReadVariableOpб
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_8/BiasAddq
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_8/Reluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_4/dropout/Constд
dropout_4/dropout/MulMuldense_8/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_4/dropout/Mul|
dropout_4/dropout/ShapeShapedense_8/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4/dropout/ShapeМ
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype020
.dropout_4/dropout/random_uniform/RandomUniformЅ
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2"
 dropout_4/dropout/GreaterEqual/yу
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2 
dropout_4/dropout/GreaterEqualъ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_4/dropout/CastБ
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_4/dropout/Mul_1д
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02
dense_9/MatMul/ReadVariableOpа
dense_9/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_9/MatMulц
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOpА
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_9/BiasAddl
IdentityIdentitydense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ:::::P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
╦
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_13166

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ѕ
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_13402

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╬
ф
B__inference_dense_9_layer_call_and_return_conditional_losses_13189

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╬
ф
B__inference_dense_9_layer_call_and_return_conditional_losses_13427

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ќ
E
)__inference_dropout_4_layer_call_fn_13417

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_131662
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
║
§
G__inference_sequential_4_layer_call_and_return_conditional_losses_13221
dense_8_input
dense_8_13209
dense_8_13211
dense_9_13215
dense_9_13217
identityѕбdense_8/StatefulPartitionedCallбdense_9/StatefulPartitionedCallћ
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_13209dense_8_13211*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_131332!
dense_8/StatefulPartitionedCallщ
dropout_4/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_131662
dropout_4/PartitionedCallе
dense_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_9_13215dense_9_13217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_131892!
dense_9/StatefulPartitionedCall└
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:W S
(
_output_shapes
:         љ
'
_user_specified_namedense_8_input
║
д
,__inference_sequential_4_layer_call_fn_13250
dense_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_132392
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:         љ
'
_user_specified_namedense_8_input
░
ф
B__inference_dense_8_layer_call_and_return_conditional_losses_13381

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
љђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         љ:::P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
Ў3
К
__inference__traced_save_13522
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_885541646fc54b7da5d4d8291032d13b/part2	
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameм
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*С

value┌
BО
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names┤
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesО
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*Ќ
_input_shapesЁ
ѓ: :
љђ:ђ:	ђ
:
: : : : : : : : : :
љђ:ђ:	ђ
:
:
љђ:ђ:	ђ
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
љђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
љђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ
: 

_output_shapes
:
:&"
 
_output_shapes
:
љђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ
: 

_output_shapes
:
:

_output_shapes
: 
Ц
Ъ
,__inference_sequential_4_layer_call_fn_13357

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_132392
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
Ц
Ъ
,__inference_sequential_4_layer_call_fn_13370

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_132672
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
Ѕ
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_13161

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
║
д
,__inference_sequential_4_layer_call_fn_13278
dense_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_132672
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:         љ
'
_user_specified_namedense_8_input
Ж
А
G__inference_sequential_4_layer_call_and_return_conditional_losses_13206
dense_8_input
dense_8_13144
dense_8_13146
dense_9_13200
dense_9_13202
identityѕбdense_8/StatefulPartitionedCallбdense_9/StatefulPartitionedCallб!dropout_4/StatefulPartitionedCallћ
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_13144dense_8_13146*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_131332!
dense_8/StatefulPartitionedCallЉ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_131612#
!dropout_4/StatefulPartitionedCall░
dense_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_9_13200dense_9_13202*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_131892!
dense_9/StatefulPartitionedCallС
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:W S
(
_output_shapes
:         љ
'
_user_specified_namedense_8_input
б
b
)__inference_dropout_4_layer_call_fn_13412

inputs
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_131612
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
зY
К

!__inference__traced_restore_13595
file_prefix#
assignvariableop_dense_8_kernel#
assignvariableop_1_dense_8_bias%
!assignvariableop_2_dense_9_kernel#
assignvariableop_3_dense_9_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_1-
)assignvariableop_13_adam_dense_8_kernel_m+
'assignvariableop_14_adam_dense_8_bias_m-
)assignvariableop_15_adam_dense_9_kernel_m+
'assignvariableop_16_adam_dense_9_bias_m-
)assignvariableop_17_adam_dense_8_kernel_v+
'assignvariableop_18_adam_dense_8_bias_v-
)assignvariableop_19_adam_dense_9_kernel_v+
'assignvariableop_20_adam_dense_9_bias_v
identity_22ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9п
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*С

value┌
BО
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names║
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityъ
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ц
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2д
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ц
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4А
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Б
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Б
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7б
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ф
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ю
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10А
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Б
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Б
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13▒
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_dense_8_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14»
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_dense_8_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15▒
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_9_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16»
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_9_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17▒
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_8_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18»
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_8_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19▒
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_9_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20»
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_9_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpг
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21Ъ
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*i
_input_shapesX
V: :::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
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
д
г
 __inference__wrapped_model_13118
dense_8_input7
3sequential_4_dense_8_matmul_readvariableop_resource8
4sequential_4_dense_8_biasadd_readvariableop_resource7
3sequential_4_dense_9_matmul_readvariableop_resource8
4sequential_4_dense_9_biasadd_readvariableop_resource
identityѕ╬
*sequential_4/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
љђ*
dtype02,
*sequential_4/dense_8/MatMul/ReadVariableOp║
sequential_4/dense_8/MatMulMatMuldense_8_input2sequential_4/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_8/MatMul╠
+sequential_4/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+sequential_4/dense_8/BiasAdd/ReadVariableOpо
sequential_4/dense_8/BiasAddBiasAdd%sequential_4/dense_8/MatMul:product:03sequential_4/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_8/BiasAddў
sequential_4/dense_8/ReluRelu%sequential_4/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_8/Reluф
sequential_4/dropout_4/IdentityIdentity'sequential_4/dense_8/Relu:activations:0*
T0*(
_output_shapes
:         ђ2!
sequential_4/dropout_4/Identity═
*sequential_4/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_9_matmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02,
*sequential_4/dense_9/MatMul/ReadVariableOpн
sequential_4/dense_9/MatMulMatMul(sequential_4/dropout_4/Identity:output:02sequential_4/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
sequential_4/dense_9/MatMul╦
+sequential_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_4/dense_9/BiasAdd/ReadVariableOpН
sequential_4/dense_9/BiasAddBiasAdd%sequential_4/dense_9/MatMul:product:03sequential_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
sequential_4/dense_9/BiasAddy
IdentityIdentity%sequential_4/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ:::::W S
(
_output_shapes
:         љ
'
_user_specified_namedense_8_input
┌
|
'__inference_dense_9_layer_call_fn_13436

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_131892
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╔
ў
G__inference_sequential_4_layer_call_and_return_conditional_losses_13344

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityѕД
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
љђ*
dtype02
dense_8/MatMul/ReadVariableOpї
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_8/MatMulЦ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_8/BiasAdd/ReadVariableOpб
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_8/BiasAddq
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_8/ReluЃ
dropout_4/IdentityIdentitydense_8/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_4/Identityд
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02
dense_9/MatMul/ReadVariableOpа
dense_9/MatMulMatMuldropout_4/Identity:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_9/MatMulц
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOpА
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_9/BiasAddl
IdentityIdentitydense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ:::::P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
Н
џ
G__inference_sequential_4_layer_call_and_return_conditional_losses_13239

inputs
dense_8_13227
dense_8_13229
dense_9_13233
dense_9_13235
identityѕбdense_8/StatefulPartitionedCallбdense_9/StatefulPartitionedCallб!dropout_4/StatefulPartitionedCallЇ
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_13227dense_8_13229*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_131332!
dense_8/StatefulPartitionedCallЉ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_131612#
!dropout_4/StatefulPartitionedCall░
dense_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_9_13233dense_9_13235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_131892!
dense_9/StatefulPartitionedCallС
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
╦
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_13407

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
і
Ю
#__inference_signature_wrapper_13301
dense_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__wrapped_model_131182
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:         љ
'
_user_specified_namedense_8_input
▄
|
'__inference_dense_8_layer_call_fn_13390

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_131332
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         љ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
Ц
Ш
G__inference_sequential_4_layer_call_and_return_conditional_losses_13267

inputs
dense_8_13255
dense_8_13257
dense_9_13261
dense_9_13263
identityѕбdense_8/StatefulPartitionedCallбdense_9/StatefulPartitionedCallЇ
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_13255dense_8_13257*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_131332!
dense_8/StatefulPartitionedCallщ
dropout_4/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_131662
dropout_4/PartitionedCallе
dense_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_9_13261dense_9_13263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_131892!
dense_9/StatefulPartitionedCall└
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         љ::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
░
ф
B__inference_dense_8_layer_call_and_return_conditional_losses_13133

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
љђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         љ:::P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*и
serving_defaultБ
H
dense_8_input7
serving_default_dense_8_input:0         љ;
dense_90
StatefulPartitionedCall:0         
tensorflow/serving/predict:▒~
о
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
F_default_save_signature
*G&call_and_return_all_conditional_losses
H__call__"░
_tf_keras_sequentialЉ{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_8_input"}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": [{"class_name": "SparseCategoricalAccuracy", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
У


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*I&call_and_return_all_conditional_losses
J__call__"├
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
т
trainable_variables
	variables
regularization_losses
	keras_api
*K&call_and_return_all_conditional_losses
L__call__"о
_tf_keras_layer╝{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
З

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*M&call_and_return_all_conditional_losses
N__call__"¤
_tf_keras_layerх{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
Џ
iter

beta_1

beta_2
	decay
learning_rate
m>m?m@mA
vBvCvDvE"
	optimizer
<

0
1
2
3"
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
trainable_variables
	variables
layer_metrics

 layers
!non_trainable_variables
regularization_losses
"metrics
#layer_regularization_losses
H__call__
F_default_save_signature
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
,
Oserving_default"
signature_map
": 
љђ2dense_8/kernel
:ђ2dense_8/bias
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
trainable_variables
	variables
$layer_metrics

%layers
&non_trainable_variables
regularization_losses
'metrics
(layer_regularization_losses
J__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
trainable_variables
	variables
)layer_metrics

*layers
+non_trainable_variables
regularization_losses
,metrics
-layer_regularization_losses
L__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
!:	ђ
2dense_9/kernel
:
2dense_9/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
trainable_variables
	variables
.layer_metrics

/layers
0non_trainable_variables
regularization_losses
1metrics
2layer_regularization_losses
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╗
	5total
	6count
7	variables
8	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Ј
	9total
	:count
;
_fn_kwargs
<	variables
=	keras_api"╚
_tf_keras_metricГ{"class_name": "SparseCategoricalAccuracy", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
.
50
61"
trackable_list_wrapper
-
7	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
90
:1"
trackable_list_wrapper
-
<	variables"
_generic_user_object
':%
љђ2Adam/dense_8/kernel/m
 :ђ2Adam/dense_8/bias/m
&:$	ђ
2Adam/dense_9/kernel/m
:
2Adam/dense_9/bias/m
':%
љђ2Adam/dense_8/kernel/v
 :ђ2Adam/dense_8/bias/v
&:$	ђ
2Adam/dense_9/kernel/v
:
2Adam/dense_9/bias/v
т2Р
 __inference__wrapped_model_13118й
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *-б*
(і%
dense_8_input         љ
Ж2у
G__inference_sequential_4_layer_call_and_return_conditional_losses_13326
G__inference_sequential_4_layer_call_and_return_conditional_losses_13221
G__inference_sequential_4_layer_call_and_return_conditional_losses_13344
G__inference_sequential_4_layer_call_and_return_conditional_losses_13206└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
■2ч
,__inference_sequential_4_layer_call_fn_13250
,__inference_sequential_4_layer_call_fn_13357
,__inference_sequential_4_layer_call_fn_13278
,__inference_sequential_4_layer_call_fn_13370└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
В2ж
B__inference_dense_8_layer_call_and_return_conditional_losses_13381б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_8_layer_call_fn_13390б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
к2├
D__inference_dropout_4_layer_call_and_return_conditional_losses_13402
D__inference_dropout_4_layer_call_and_return_conditional_losses_13407┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
љ2Ї
)__inference_dropout_4_layer_call_fn_13412
)__inference_dropout_4_layer_call_fn_13417┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
В2ж
B__inference_dense_9_layer_call_and_return_conditional_losses_13427б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_9_layer_call_fn_13436б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
8B6
#__inference_signature_wrapper_13301dense_8_inputќ
 __inference__wrapped_model_13118r
7б4
-б*
(і%
dense_8_input         љ
ф "1ф.
,
dense_9!і
dense_9         
ц
B__inference_dense_8_layer_call_and_return_conditional_losses_13381^
0б-
&б#
!і
inputs         љ
ф "&б#
і
0         ђ
џ |
'__inference_dense_8_layer_call_fn_13390Q
0б-
&б#
!і
inputs         љ
ф "і         ђБ
B__inference_dense_9_layer_call_and_return_conditional_losses_13427]0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         

џ {
'__inference_dense_9_layer_call_fn_13436P0б-
&б#
!і
inputs         ђ
ф "і         
д
D__inference_dropout_4_layer_call_and_return_conditional_losses_13402^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ д
D__inference_dropout_4_layer_call_and_return_conditional_losses_13407^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ ~
)__inference_dropout_4_layer_call_fn_13412Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђ~
)__inference_dropout_4_layer_call_fn_13417Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђ╣
G__inference_sequential_4_layer_call_and_return_conditional_losses_13206n
?б<
5б2
(і%
dense_8_input         љ
p

 
ф "%б"
і
0         

џ ╣
G__inference_sequential_4_layer_call_and_return_conditional_losses_13221n
?б<
5б2
(і%
dense_8_input         љ
p 

 
ф "%б"
і
0         

џ ▓
G__inference_sequential_4_layer_call_and_return_conditional_losses_13326g
8б5
.б+
!і
inputs         љ
p

 
ф "%б"
і
0         

џ ▓
G__inference_sequential_4_layer_call_and_return_conditional_losses_13344g
8б5
.б+
!і
inputs         љ
p 

 
ф "%б"
і
0         

џ Љ
,__inference_sequential_4_layer_call_fn_13250a
?б<
5б2
(і%
dense_8_input         љ
p

 
ф "і         
Љ
,__inference_sequential_4_layer_call_fn_13278a
?б<
5б2
(і%
dense_8_input         љ
p 

 
ф "і         
і
,__inference_sequential_4_layer_call_fn_13357Z
8б5
.б+
!і
inputs         љ
p

 
ф "і         
і
,__inference_sequential_4_layer_call_fn_13370Z
8б5
.б+
!і
inputs         љ
p 

 
ф "і         
Ф
#__inference_signature_wrapper_13301Ѓ
HбE
б 
>ф;
9
dense_8_input(і%
dense_8_input         љ"1ф.
,
dense_9!і
dense_9         
