// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_SUPERVISED_CC
#define NETKET_SUPERVISED_CC

#include <memory>

#include "Optimizer/optimizer.hpp"
#include "vmc.hpp"

namespace netket {

class Supervised {
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

  int batchsize_ = 10;

  Eigen::MatrixXd inputs_;
  Eigen::VectorXcd targets_;

 public:
  explicit Supervised(const json &supervised_pars) {
    // Relevant parameters for supervised learning
    // is stored in supervised_pars.
    CheckFieldExists(supervised_pars, "Supervised");
    const std::string loss_name =
        FieldVal(supervised_pars["Supervised"], "Loss", "Supervised");

    // Input data is encoded in Json file with name "InputFilename".
    auto data_json =
        ReadJsonFromFile(supervised_pars["Supervised"]["InputFilename"]);
    using DataType = Data<double>;
    DataType data(data_json, supervised_pars);

    // Make a machine using the Hilbert space extracted from
    // the data.
    using MachineType = Machine<std::complex<double>>;
    MachineType machine(data.GetHilbert(), supervised_pars);

    if (loss_name == "Overlap") {
      // To do:
      // Check whether we need more advance sampler, i.e. exchange or hop,
      // which are constructed with Graph object provided?
      Sampler<MachineType> sampler(machine, supervised_pars);
      Optimizer optimizer(supervised_pars);

      // To do:
      // Consider adding function (Grad, Init, Run_Supervised) in VMC class,
      // So we do not need to copy all the function VMC class again.
      std::cout << " sampler created, optimizer created \n";
      SupervisedVariationalMonteCarlo vmc(data, sampler, optimizer,
                                          supervised_pars);
      vmc.Run_Supervised();
    } else if (loss_name == "MSE") {
      inputs_.resize(batchsize_, machine.Nvisible());
      targets_.resize(batchsize_);

      VectorType gradC(machine.Npar());

      std::cout << "Running epochs" << std::endl;
      for (int epoch = 0; epoch < 10; ++epoch) {
        int number_of_batches = floor(data.Ndata() / batchsize_);

        for (int iteration = 0; iteration < number_of_batches; ++iteration) {
          // Generate a batch from the data
          data.GenerateBatch(batchsize_, inputs_, targets_);

          // Compute the gradients
          gradC.setZero();
          for (int x = 0; x < batchsize_; ++x) {
            Eigen::VectorXd config(inputs_.row(x));

            std::complex<double> value = machine.LogVal(config);
            auto partial_gradient = machine.DerLog(config);
            gradC = gradC + partial_gradient * (value - targets_(x));

            // std::cout << "Current gradient: " << gradC << std::endl;
          }

          // Update the parameters
          double alpha = 1e-3;
          machine.SetParameters(machine.GetParameters() - alpha * gradC);
        }
      }

    } else {
      std::stringstream s;
      s << "Unknown Supervised loss: " << loss_name;
      throw InvalidInputError(s.str());
    }
  }
};

}  // namespace netket

#endif
