@0x833a5be3c1c85717;

struct Tensor {
  name @0 :Text;
  data @1 :Data;
}

struct TensorArray {
  array @0: List(Tensor);
}

struct DataInfo {
  start @0: Int32;
  end   @1: Int32;
}

struct Msg {
  msg @0 :Text;
}

interface Worker {
  loadData @0 (info: DataInfo) -> (msg: Msg);
  runStep  @1 (ins: TensorArray) -> (outs: TensorArray);
  endRun @2() -> (msg: Msg);
}