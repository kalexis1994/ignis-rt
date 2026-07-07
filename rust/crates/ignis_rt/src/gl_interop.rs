//! OpenGL interop — draws the Vulkan-rendered frame directly into Blender's GL viewport
//! (zero-copy) instead of the CPU float readback + GPUTexture upload fallback.
//!
//! Port of the GL half of the C++ `src/vk/vk_interop.cpp`: the renderer exports its shared
//! R8G8B8A8 images as OPAQUE_WIN32 NT handles; here we import them as GL memory objects
//! (GL_EXT_memory_object_win32), wrap them in GL textures, and blit a fullscreen triangle
//! into whatever framebuffer Blender has bound during `view_draw` (its GL context is current
//! on this thread at that point, which is what makes the raw-GL approach legal).
//!
//! Sync model: render() is synchronous (fence-waited) and double-buffered on the Vulkan side,
//! so by the time the addon calls draw_gl the completed buffer is fully written — same model
//! as the C++ (which used vkQueueWaitIdle). State touched in the GL context is saved/restored
//! exactly like the C++ DrawGL so Blender's own GPU state tracker never notices us.

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::os::raw::c_char;
use std::sync::Mutex;

use crate::log::log;

// ── Win32 (kernel32 + opengl32) ──────────────────────────────────────────────────────────

#[link(name = "kernel32")]
extern "system" {
    fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn GetCurrentProcess() -> *mut c_void;
    fn DuplicateHandle(
        src_process: *mut c_void, src: *mut c_void, dst_process: *mut c_void,
        dst: *mut *mut c_void, access: u32, inherit: i32, options: u32,
    ) -> i32;
    fn CloseHandle(h: *mut c_void) -> i32;
}
const DUPLICATE_SAME_ACCESS: u32 = 0x2;

// ── GL types + constants (only what we use) ──────────────────────────────────────────────

type GLenum = u32;
type GLuint = u32;
type GLint = i32;
type GLsizei = i32;
type GLboolean = u8;
type GLuint64 = u64;

const GL_TEXTURE_2D: GLenum = 0x0DE1;
const GL_RGBA8: GLenum = 0x8058;
const GL_TEXTURE_MIN_FILTER: GLenum = 0x2801;
const GL_TEXTURE_MAG_FILTER: GLenum = 0x2800;
const GL_TEXTURE_WRAP_S: GLenum = 0x2802;
const GL_TEXTURE_WRAP_T: GLenum = 0x2803;
const GL_NEAREST: GLint = 0x2600;
const GL_CLAMP_TO_EDGE: GLint = 0x812F;
const GL_BLEND: GLenum = 0x0BE2;
const GL_DEPTH_TEST: GLenum = 0x0B71;
const GL_SCISSOR_TEST: GLenum = 0x0C11;
const GL_FRAMEBUFFER_SRGB: GLenum = 0x8DB9;
const GL_TRIANGLES: GLenum = 0x0004;
const GL_FRAGMENT_SHADER: GLenum = 0x8B30;
const GL_VERTEX_SHADER: GLenum = 0x8B31;
const GL_COMPILE_STATUS: GLenum = 0x8B81;
const GL_LINK_STATUS: GLenum = 0x8B82;
const GL_CURRENT_PROGRAM: GLenum = 0x8B8D;
const GL_VERTEX_ARRAY_BINDING: GLenum = 0x85B5;
const GL_ACTIVE_TEXTURE: GLenum = 0x84E0;
const GL_TEXTURE_BINDING_2D: GLenum = 0x8069;
const GL_TEXTURE0: GLenum = 0x84C0;
const GL_HANDLE_TYPE_OPAQUE_WIN32_EXT: GLenum = 0x9587;

// ── GL function pointers ─────────────────────────────────────────────────────────────────

macro_rules! gl_fns {
    ($( $name:ident : fn($($arg:ty),*) $(-> $ret:ty)? ),+ $(,)?) => {
        #[derive(Default)]
        struct GlFns {
            $( $name: Option<unsafe extern "system" fn($($arg),*) $(-> $ret)?>, )+
        }
    };
}

gl_fns! {
    GetIntegerv: fn(GLenum, *mut GLint),
    BindTexture: fn(GLenum, GLuint),
    GenTextures: fn(GLsizei, *mut GLuint),
    DeleteTextures: fn(GLsizei, *const GLuint),
    TexParameteri: fn(GLenum, GLenum, GLint),
    Enable: fn(GLenum),
    Disable: fn(GLenum),
    IsEnabled: fn(GLenum) -> GLboolean,
    DrawArrays: fn(GLenum, GLint, GLsizei),
    GetError: fn() -> GLenum,
    CreateShader: fn(GLenum) -> GLuint,
    ShaderSource: fn(GLuint, GLsizei, *const *const c_char, *const GLint),
    CompileShader: fn(GLuint),
    GetShaderiv: fn(GLuint, GLenum, *mut GLint),
    GetShaderInfoLog: fn(GLuint, GLsizei, *mut GLsizei, *mut c_char),
    CreateProgram: fn() -> GLuint,
    AttachShader: fn(GLuint, GLuint),
    LinkProgram: fn(GLuint),
    GetProgramiv: fn(GLuint, GLenum, *mut GLint),
    GetProgramInfoLog: fn(GLuint, GLsizei, *mut GLsizei, *mut c_char),
    UseProgram: fn(GLuint),
    DeleteShader: fn(GLuint),
    DeleteProgram: fn(GLuint),
    GetUniformLocation: fn(GLuint, *const c_char) -> GLint,
    Uniform1i: fn(GLint, GLint),
    ActiveTexture: fn(GLenum),
    GenVertexArrays: fn(GLsizei, *mut GLuint),
    BindVertexArray: fn(GLuint),
    DeleteVertexArrays: fn(GLsizei, *const GLuint),
    CreateMemoryObjectsEXT: fn(GLsizei, *mut GLuint),
    DeleteMemoryObjectsEXT: fn(GLsizei, *const GLuint),
    TexStorageMem2DEXT: fn(GLenum, GLsizei, GLenum, GLsizei, GLsizei, GLuint, GLuint64),
    ImportMemoryWin32HandleEXT: fn(GLuint, GLuint64, GLenum, *mut c_void),
}

// Fullscreen triangle, Y-flipped (Vulkan rows are top-down; GL samples bottom-up).
const VS_SRC: &str = "#version 330 core\n\
out vec2 vUV;\n\
void main() {\n\
    float x = (gl_VertexID == 1) ? 3.0 : -1.0;\n\
    float y = (gl_VertexID == 2) ? 3.0 : -1.0;\n\
    vUV = vec2((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);\n\
    gl_Position = vec4(x, y, 0.0, 1.0);\n\
}\n\0";
const FS_SRC: &str = "#version 330 core\n\
in vec2 vUV;\n\
out vec4 fragColor;\n\
uniform sampler2D uTexture;\n\
void main() { fragColor = texture(uTexture, vUV); }\n\0";

struct GlState {
    fns: GlFns,
    loaded: bool,
    disabled: bool,        // a hard init failure happened — never retry (addon falls back)
    generation: u64,       // which renderer generation the imports belong to
    mem_objects: [GLuint; 2],
    textures: [GLuint; 2],
    program: GLuint,
    vao: GLuint,
    ready: bool,
}

// Blender calls view_draw from one thread, but Rust statics need sync anyway.
static GL: Mutex<Option<GlState>> = Mutex::new(None);

unsafe fn load_fns(st: &mut GlState) -> bool {
    if st.loaded {
        return true;
    }
    let h_gl = GetModuleHandleA(b"opengl32.dll\0".as_ptr() as *const c_char);
    if h_gl.is_null() {
        log("[gl-interop] opengl32.dll not loaded in process (Blender on a non-GL backend?)");
        return false;
    }
    let wgl_get_proc: Option<unsafe extern "system" fn(*const c_char) -> *mut c_void> =
        std::mem::transmute(GetProcAddress(h_gl, b"wglGetProcAddress\0".as_ptr() as *const c_char));
    let Some(wgl_get_proc) = wgl_get_proc else {
        log("[gl-interop] wglGetProcAddress not found");
        return false;
    };
    // Core 1.x entry points live in opengl32.dll; >=2.0 and extensions come from the ICD via wgl.
    unsafe fn load(h_gl: *mut c_void, wgl: unsafe extern "system" fn(*const c_char) -> *mut c_void,
                   name: &[u8]) -> *mut c_void {
        let p = GetProcAddress(h_gl, name.as_ptr() as *const c_char);
        if !p.is_null() { return p; }
        wgl(name.as_ptr() as *const c_char)
    }
    macro_rules! ld {
        ($field:ident, $name:literal) => {
            st.fns.$field = std::mem::transmute(load(h_gl, wgl_get_proc, $name));
        };
    }
    ld!(GetIntegerv, b"glGetIntegerv\0");
    ld!(BindTexture, b"glBindTexture\0");
    ld!(GenTextures, b"glGenTextures\0");
    ld!(DeleteTextures, b"glDeleteTextures\0");
    ld!(TexParameteri, b"glTexParameteri\0");
    ld!(Enable, b"glEnable\0");
    ld!(Disable, b"glDisable\0");
    ld!(IsEnabled, b"glIsEnabled\0");
    ld!(DrawArrays, b"glDrawArrays\0");
    ld!(GetError, b"glGetError\0");
    ld!(CreateShader, b"glCreateShader\0");
    ld!(ShaderSource, b"glShaderSource\0");
    ld!(CompileShader, b"glCompileShader\0");
    ld!(GetShaderiv, b"glGetShaderiv\0");
    ld!(GetShaderInfoLog, b"glGetShaderInfoLog\0");
    ld!(CreateProgram, b"glCreateProgram\0");
    ld!(AttachShader, b"glAttachShader\0");
    ld!(LinkProgram, b"glLinkProgram\0");
    ld!(GetProgramiv, b"glGetProgramiv\0");
    ld!(GetProgramInfoLog, b"glGetProgramInfoLog\0");
    ld!(UseProgram, b"glUseProgram\0");
    ld!(DeleteShader, b"glDeleteShader\0");
    ld!(DeleteProgram, b"glDeleteProgram\0");
    ld!(GetUniformLocation, b"glGetUniformLocation\0");
    ld!(Uniform1i, b"glUniform1i\0");
    ld!(ActiveTexture, b"glActiveTexture\0");
    ld!(GenVertexArrays, b"glGenVertexArrays\0");
    ld!(BindVertexArray, b"glBindVertexArray\0");
    ld!(DeleteVertexArrays, b"glDeleteVertexArrays\0");
    ld!(CreateMemoryObjectsEXT, b"glCreateMemoryObjectsEXT\0");
    ld!(DeleteMemoryObjectsEXT, b"glDeleteMemoryObjectsEXT\0");
    ld!(TexStorageMem2DEXT, b"glTexStorageMem2DEXT\0");
    ld!(ImportMemoryWin32HandleEXT, b"glImportMemoryWin32HandleEXT\0");

    let f = &st.fns;
    if f.GetIntegerv.is_none() || f.BindTexture.is_none() || f.GenTextures.is_none()
        || f.DrawArrays.is_none() || f.CreateShader.is_none() || f.UseProgram.is_none()
        || f.GenVertexArrays.is_none() || f.BindVertexArray.is_none()
        || f.CreateMemoryObjectsEXT.is_none() || f.ImportMemoryWin32HandleEXT.is_none()
        || f.TexStorageMem2DEXT.is_none()
    {
        log("[gl-interop] required GL functions missing (no GL_EXT_memory_object_win32?)");
        return false;
    }
    st.loaded = true;
    true
}

unsafe fn compile_shader(f: &GlFns, kind: GLenum, src: &str) -> GLuint {
    let shader = (f.CreateShader.unwrap())(kind);
    let p = src.as_ptr() as *const c_char;
    (f.ShaderSource.unwrap())(shader, 1, &p, std::ptr::null());
    (f.CompileShader.unwrap())(shader);
    let mut ok: GLint = 0;
    (f.GetShaderiv.unwrap())(shader, GL_COMPILE_STATUS, &mut ok);
    if ok == 0 {
        let mut buf = [0i8; 512];
        (f.GetShaderInfoLog.unwrap())(shader, 512, std::ptr::null_mut(), buf.as_mut_ptr());
        let msg = std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned();
        log(&format!("[gl-interop] shader compile error: {msg}"));
        (f.DeleteShader.unwrap())(shader);
        return 0;
    }
    shader
}

/// Tear down the GL objects for the current imports (only safe with a GL context current —
/// checked via wglGetCurrentContext, like the C++ ShutdownGL).
unsafe fn shutdown_gl_objects(st: &mut GlState) {
    let h_gl = GetModuleHandleA(b"opengl32.dll\0".as_ptr() as *const c_char);
    let mut can_call = false;
    if !h_gl.is_null() && st.loaded {
        let get_ctx: Option<unsafe extern "system" fn() -> *mut c_void> =
            std::mem::transmute(GetProcAddress(h_gl, b"wglGetCurrentContext\0".as_ptr() as *const c_char));
        can_call = get_ctx.map(|f| !f().is_null()).unwrap_or(false);
    }
    if can_call {
        let f = &st.fns;
        if st.vao != 0 { (f.DeleteVertexArrays.unwrap())(1, &st.vao); }
        if st.program != 0 { (f.DeleteProgram.unwrap())(st.program); }
        for i in 0..2 {
            if st.textures[i] != 0 { (f.DeleteTextures.unwrap())(1, &st.textures[i]); }
            if st.mem_objects[i] != 0 { (f.DeleteMemoryObjectsEXT.unwrap())(1, &st.mem_objects[i]); }
        }
    }
    st.vao = 0;
    st.program = 0;
    st.textures = [0; 2];
    st.mem_objects = [0; 2];
    st.ready = false;
}

/// Import the renderer's shared images (NT handles) as GL textures + build the draw program.
/// `generation` identifies the renderer instance — a new renderer (resize) re-imports.
pub fn ensure_ready(handles: [*mut c_void; 2], alloc_sizes: [u64; 2],
                    width: u32, height: u32, generation: u64) -> bool {
    let mut guard = GL.lock().unwrap();
    let st = guard.get_or_insert_with(|| GlState {
        fns: GlFns::default(), loaded: false, disabled: false, generation: 0,
        mem_objects: [0; 2], textures: [0; 2], program: 0, vao: 0, ready: false,
    });
    if st.disabled {
        return false;
    }
    if st.ready && st.generation == generation {
        return true;
    }
    unsafe {
        if st.ready {
            shutdown_gl_objects(st); // stale imports from a previous renderer generation
        }
        if !load_fns(st) {
            st.disabled = true; // context has no GL / no extension — permanent fallback
            return false;
        }
        let f_getError = st.fns.GetError.unwrap();
        while f_getError() != 0 {}

        for i in 0..2 {
            (st.fns.CreateMemoryObjectsEXT.unwrap())(1, &mut st.mem_objects[i]);
            if st.mem_objects[i] == 0 {
                log("[gl-interop] glCreateMemoryObjectsEXT failed");
                shutdown_gl_objects(st);
                st.disabled = true;
                return false;
            }
            // The GL driver consumes the handle — hand it a duplicate (C++ parity).
            let mut dup: *mut c_void = std::ptr::null_mut();
            if DuplicateHandle(GetCurrentProcess(), handles[i], GetCurrentProcess(),
                               &mut dup, 0, 0, DUPLICATE_SAME_ACCESS) == 0 {
                log("[gl-interop] DuplicateHandle failed");
                shutdown_gl_objects(st);
                st.disabled = true;
                return false;
            }
            (st.fns.ImportMemoryWin32HandleEXT.unwrap())(
                st.mem_objects[i], alloc_sizes[i], GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, dup);
            let err = f_getError();
            if err != 0 {
                log(&format!("[gl-interop] glImportMemoryWin32HandleEXT failed (GL 0x{err:X})"));
                CloseHandle(dup);
                shutdown_gl_objects(st);
                st.disabled = true;
                return false;
            }
            (st.fns.GenTextures.unwrap())(1, &mut st.textures[i]);
            (st.fns.BindTexture.unwrap())(GL_TEXTURE_2D, st.textures[i]);
            (st.fns.TexParameteri.unwrap())(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            (st.fns.TexParameteri.unwrap())(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            (st.fns.TexParameteri.unwrap())(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            (st.fns.TexParameteri.unwrap())(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            (st.fns.TexStorageMem2DEXT.unwrap())(
                GL_TEXTURE_2D, 1, GL_RGBA8, width as GLsizei, height as GLsizei, st.mem_objects[i], 0);
            let err = f_getError();
            (st.fns.BindTexture.unwrap())(GL_TEXTURE_2D, 0);
            if err != 0 {
                log(&format!("[gl-interop] glTexStorageMem2DEXT failed (GL 0x{err:X})"));
                shutdown_gl_objects(st);
                st.disabled = true;
                return false;
            }
        }

        if st.program == 0 {
            let vs = compile_shader(&st.fns, GL_VERTEX_SHADER, VS_SRC);
            let fs = compile_shader(&st.fns, GL_FRAGMENT_SHADER, FS_SRC);
            if vs == 0 || fs == 0 {
                shutdown_gl_objects(st);
                st.disabled = true;
                return false;
            }
            let f = &st.fns;
            st.program = (f.CreateProgram.unwrap())();
            (f.AttachShader.unwrap())(st.program, vs);
            (f.AttachShader.unwrap())(st.program, fs);
            (f.LinkProgram.unwrap())(st.program);
            (f.DeleteShader.unwrap())(vs);
            (f.DeleteShader.unwrap())(fs);
            let mut linked: GLint = 0;
            (f.GetProgramiv.unwrap())(st.program, GL_LINK_STATUS, &mut linked);
            if linked == 0 {
                log("[gl-interop] shader program link failed");
                shutdown_gl_objects(st);
                st.disabled = true;
                return false;
            }
            (f.UseProgram.unwrap())(st.program);
            let loc = (f.GetUniformLocation.unwrap())(st.program, b"uTexture\0".as_ptr() as *const c_char);
            (f.Uniform1i.unwrap())(loc, 0);
            (f.UseProgram.unwrap())(0);
        }
        if st.vao == 0 {
            (st.fns.GenVertexArrays.unwrap())(1, &mut st.vao);
        }
    }
    st.generation = generation;
    st.ready = true;
    log(&format!("[gl-interop] ready: {width}x{height}, 2 shared buffers (zero-copy GL path active)"));
    true
}

/// Draw the completed shared buffer as a fullscreen triangle into the currently bound
/// framebuffer. State save/restore mirrors the C++ DrawGL exactly.
pub fn draw(read_idx: u32) -> bool {
    let mut guard = GL.lock().unwrap();
    let Some(st) = guard.as_mut() else { return false };
    if !st.ready || st.disabled {
        return false;
    }
    unsafe {
        let f = &st.fns;
        let gi = f.GetIntegerv.unwrap();
        let mut prev_program: GLint = 0;
        let mut prev_vao: GLint = 0;
        let mut prev_active: GLint = 0;
        let mut prev_tex: GLint = 0;
        gi(GL_CURRENT_PROGRAM, &mut prev_program);
        gi(GL_VERTEX_ARRAY_BINDING, &mut prev_vao);
        gi(GL_ACTIVE_TEXTURE, &mut prev_active);
        (f.ActiveTexture.unwrap())(GL_TEXTURE0);
        gi(GL_TEXTURE_BINDING_2D, &mut prev_tex);
        let prev_blend = (f.IsEnabled.unwrap())(GL_BLEND);
        let prev_depth = (f.IsEnabled.unwrap())(GL_DEPTH_TEST);
        let prev_scissor = (f.IsEnabled.unwrap())(GL_SCISSOR_TEST);
        let prev_srgb = (f.IsEnabled.unwrap())(GL_FRAMEBUFFER_SRGB);

        (f.Disable.unwrap())(GL_BLEND);
        (f.Disable.unwrap())(GL_DEPTH_TEST);
        (f.Disable.unwrap())(GL_SCISSOR_TEST);
        // The shared texture already holds display-ready SDR values — if Blender's context has
        // GL_FRAMEBUFFER_SRGB on, GL would encode AGAIN on write and wash the image out.
        (f.Disable.unwrap())(GL_FRAMEBUFFER_SRGB);

        let tex = if st.textures[read_idx as usize] != 0 { st.textures[read_idx as usize] } else { st.textures[0] };
        (f.UseProgram.unwrap())(st.program);
        (f.BindVertexArray.unwrap())(st.vao);
        (f.ActiveTexture.unwrap())(GL_TEXTURE0);
        (f.BindTexture.unwrap())(GL_TEXTURE_2D, tex);
        (f.DrawArrays.unwrap())(GL_TRIANGLES, 0, 3);

        (f.UseProgram.unwrap())(prev_program as GLuint);
        (f.BindVertexArray.unwrap())(prev_vao as GLuint);
        (f.ActiveTexture.unwrap())(GL_TEXTURE0);
        (f.BindTexture.unwrap())(GL_TEXTURE_2D, prev_tex as GLuint);
        (f.ActiveTexture.unwrap())(prev_active as GLenum);
        let set = |on: GLboolean, cap: GLenum| {
            if on != 0 { (f.Enable.unwrap())(cap) } else { (f.Disable.unwrap())(cap) }
        };
        set(prev_blend, GL_BLEND);
        set(prev_depth, GL_DEPTH_TEST);
        set(prev_scissor, GL_SCISSOR_TEST);
        set(prev_srgb, GL_FRAMEBUFFER_SRGB);
    }
    true
}

/// Whether the GL path was hard-disabled (init failed once — the addon should keep using
/// the readback fallback without retrying).
pub fn is_disabled() -> bool {
    GL.lock().unwrap().as_ref().map(|s| s.disabled).unwrap_or(false)
}

/// Hard-disable the GL path (e.g. the Vulkan-side shared-image creation failed).
pub fn disable() {
    let mut guard = GL.lock().unwrap();
    let st = guard.get_or_insert_with(|| GlState {
        fns: GlFns::default(), loaded: false, disabled: false, generation: 0,
        mem_objects: [0; 2], textures: [0; 2], program: 0, vao: 0, ready: false,
    });
    st.disabled = true;
}

/// Close a Win32 HANDLE stored as isize (used by the renderer's Drop for the NT handles).
pub fn close_handle(h: isize) {
    if h != 0 {
        unsafe { CloseHandle(h as *mut c_void) };
    }
}

/// Called from the renderer's Drop: release GL objects if a context happens to be current.
pub fn shutdown() {
    let mut guard = GL.lock().unwrap();
    if let Some(st) = guard.as_mut() {
        unsafe { shutdown_gl_objects(st) };
    }
}
