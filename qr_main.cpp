// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poputil/TileMapping.hpp>
#include <poplin/experimental/QRFactorization.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;

int main() {

  unsigned numRows = 5;
  unsigned numCols = 5;

  // Create the DeviceManager which is used to discover devices
  auto manager = DeviceManager::createDeviceManager();

  // Attempt to attach to a single IPU:
  auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
  std::cout << "Trying to attach to IPU\n";
  auto it = std::find_if(devices.begin(), devices.end(),
                         [](Device &device) { return device.attach(); });

  if (it == devices.end()) {
    std::cerr << "Error attaching to device\n";
    return 1; // EXIT_FAILURE
  }

  auto device = std::move(*it);
  std::cout << "Attached to IPU " << device.getId() << std::endl;

  auto target = device.getTarget();

  std::cout << "Creating environment (compiling vertex programs)\n";

  Graph graph(target);

  std::cout << "Constructing compute graph and control program\n";
  // Create tensors in the graph to hold the input/output data.
  Tensor matrix = graph.addVariable(FLOAT, {numRows, numCols}, "matrix");
  Tensor A = graph.addVariable(FLOAT, {numRows, numCols}, "A");
  Tensor Q = graph.addVariable(FLOAT, {numRows, numCols}, "Q");
  
  poputil::mapTensorLinearly(graph, matrix);
  graph.setTileMapping(A,0);
  graph.setTileMapping(Q,0);

  auto hMatrix = std::vector<float>(numRows * numCols);

  for (unsigned col = 0; col < numCols; ++col) {
    for (unsigned row = 0; row < numRows; ++row) {
      hMatrix[row * numCols + col] = row + col;
    }
  }

  auto matrices = poplin::experimental::createQRFactorizationMatrices(graph, FLOAT,{numRows}, {numCols}, "matrices" );
  //PrintTensor("matrices", matrices);

  auto inStreamM = graph.addHostToDeviceFIFO("inputMatrix", FLOAT,  numRows * numCols);
  auto MatStream_0 = graph.addHostToDeviceFIFO("matrices[0]", FLOAT, numRows * numRows);
  auto MatStream_1 = graph.addHostToDeviceFIFO("matrices[1]", FLOAT, numRows * numCols);

  auto prog = Sequence({Copy(inStreamM, matrix), Copy(MatStream_0,matrices[0]), Copy(MatStream_1, matrices[1]),
                             PrintTensor("matrix", matrix), PrintTensor("matrices[0]", matrices[0]), PrintTensor("matrices[0]", matrices[1])});
 
  Engine engine(graph, prog);
  engine.load(device);
  engine.connectStream("inputMatrix", hMatrix.data());
  //engine.connectStream("inputMatrix", hMatrix.data());

 
  std::cout << "Running graph program to multiply matrix by vector\n";
  engine.run();

}