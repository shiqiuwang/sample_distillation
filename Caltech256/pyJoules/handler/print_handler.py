# MIT License
# Copyright (c) 2019, INRIA
# Copyright (c) 2019, University of Lille
# All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import reduce
from operator import add

from ..energy_trace import EnergyTrace
from .handler import EnergyHandler


class PrintHandler(EnergyHandler):

    def process(self, trace: EnergyTrace):
        """
        Print the given sample on the standard output
        """
        for sample in trace:
            begin_string = f'begin timestamp : {sample.timestamp}; tag : {sample.tag}; duration : {sample.duration}'
            energy_strings = [f'; {domain} : {value*2.778*(1e-13)}' for domain, value in sample.energy.items()]
            print(reduce(add, energy_strings, begin_string))
            #print("dram_energy:")
            #print(energy_strings.get('dram_0')+energy_strings.get('dram_1'))
