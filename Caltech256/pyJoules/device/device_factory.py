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

from typing import List, Optional
import logging
from operator import add
from . import Domain, Device
from .rapl_device import RaplDevice
try:
    from .nvidia_device import NvidiaGPUDevice
except ImportError: 
    NvidiaGPUDevice=None
    logging.warning('pynvml not found you can\'t use NVIDIA devices') 

from ..exception import NoSuchDeviceError

from functools import reduce


class DeviceFactory:

    @staticmethod
    def _gen_all_available_domains() -> List[Device]:
        available_api = [RaplDevice]
        if NvidiaGPUDevice is not None : 
            available_api.append(NvidiaGPUDevice)
        available_domains = []
        for api in available_api:
            try:
                available_domains.append(api.available_domains())
            except NoSuchDeviceError:
                pass
        flaten_available_domain_list = reduce(add, available_domains, [])
        return flaten_available_domain_list

    @staticmethod
    def create_devices(domains: Optional[Domain] = None) -> List[Device]:
        """
        Create and configure the Device instance with the given Domains

        :param domains: a list of Domain instance that as to be monitored (if None, return a list of all
                        monitorable devices)
        :return: a list of device configured with the given Domains
        :raise NoSuchDeviceError: if a domain depend on a device that doesn't exist on the current machine
        :raise NoSuchDomainError: if the given domain is not available on the device
        """
        if domains is None:
            domains = DeviceFactory._gen_all_available_domains()

        grouped_domains = {}
        for device_type, domain in map(lambda d: (d.get_device_type(), d), domains):
            if device_type in grouped_domains:
                grouped_domains[device_type].append(domain)
            else:
                grouped_domains[device_type] = [domain]

        devices = []

        for device_type in grouped_domains:
            device = device_type()
            device.configure(domains=grouped_domains[device_type])
            devices.append(device)
        return devices
