let g:VimuxUseNearest = 0

set wildignore+=*/mied.egg-info/*,*.mp4,*.tar,*.tar.bkup,*/exps/**/*,*/wandb/*,exps

function! RunTest(exp_name, run_py, gpu_id, args)
    let l:cwd = getcwd()
    let l:cmd = 'cd ' . l:cwd . '/tests/' . a:exp_name . ';'
    if executable('conda')
        let l:cmd = l:cmd . 'conda activate mied;'
    endif
    let l:cmd = l:cmd . 'CUDA_VISIBLE_DEVICES=' . a:gpu_id . ' python ' . a:run_py . ' '. a:args
    call VimuxRunCommand(l:cmd)
endfunction
