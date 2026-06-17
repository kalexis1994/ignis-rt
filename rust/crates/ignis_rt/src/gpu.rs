//! Small GPU buffer helper over ash + gpu-allocator. Used for acceleration-structure
//! inputs (vertices/indices/instances), AS storage, and scratch buffers.

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub alloc: Option<Allocation>,
    pub size: u64,
    mapped: usize, // host pointer, 0 if not host-visible
}

impl GpuBuffer {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self, String> {
        let size = size.max(1);
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&ci, None) }
            .map_err(|e| format!("create_buffer {name}: {e}"))?;
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let alloc = allocator
            .allocate(&AllocationCreateDesc {
                name,
                requirements: req,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| format!("alloc {name}: {e}"))?;
        unsafe { device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset()) }
            .map_err(|e| format!("bind {name}: {e}"))?;
        let mapped = alloc.mapped_ptr().map(|p| p.as_ptr() as usize).unwrap_or(0);
        Ok(Self { buffer, alloc: Some(alloc), size, mapped })
    }

    pub fn device_address(&self, device: &ash::Device) -> vk::DeviceAddress {
        unsafe {
            device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(self.buffer),
            )
        }
    }

    /// Copy bytes into a host-visible buffer (no-op if the buffer isn't mapped).
    pub fn write_bytes(&self, data: &[u8]) {
        if self.mapped != 0 {
            let n = data.len().min(self.size as usize);
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped as *mut u8, n) };
        }
    }

    pub fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe { device.destroy_buffer(self.buffer, None) };
        if let Some(a) = self.alloc.take() {
            let _ = allocator.free(a);
        }
    }
}

/// Reinterpret a slice as bytes (for host-visible uploads).
pub fn as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
    }
}
