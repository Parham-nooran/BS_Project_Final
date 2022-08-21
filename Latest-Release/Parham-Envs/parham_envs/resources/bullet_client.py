"""A wrapper for pybullet to manage different clients."""
from __future__ import absolute_import
from __future__ import division
import os
import functools
import inspect
import pybullet


class BulletClient(object):

  def __init__(self, connection_mode=None, hostName=None, options=''):
    
    self._shapes = {}
    self._pid = os.getpid()
    if connection_mode is None:
      self._client = pybullet.connect(pybullet.SHARED_MEMORY, options=options)
      if self._client >= 0:
        return
      else:
        connection_mode = pybullet.DIRECT
    if hostName is None:
        self._client = pybullet.connect(connection_mode, options=options)
    else:
        self._client = pybullet.connect(connection_mode, hostName=hostName, options=options)

  def __del__(self):
    
    if self._client>=0 and self._pid == os.getpid():
      try:
        pybullet.disconnect(physicsClientId=self._client)
        self._client = -1
      except pybullet.error:
        pass

  def __getattr__(self, name):
    
    attribute = getattr(pybullet, name)
    if inspect.isbuiltin(attribute):
      attribute = functools.partial(attribute, physicsClientId=self._client)
    if name=="disconnect":
      self._client = -1 
    return attribute