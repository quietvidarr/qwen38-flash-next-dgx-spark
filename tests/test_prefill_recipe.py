"""Public opt-in, memory, patch and installed-source contracts. CPU only."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PrefillRecipe(unittest.TestCase):
    def test_switch_default_off_and_explicit_opt_in(self):
        serve = load('serve')
        for enabled in ['0', '1']:
            with patch.dict(os.environ, {'QWEN_PREFILL_KERNEL': enabled}), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(serve.main(['--model','/model','--vision-weights','/tower','--print-config']),0)
            config = json.loads(out.getvalue())
            self.assertEqual(config['prompt_expert_kernels'], enabled == '1')
            self.assertEqual(config['engine']['max_len'],262144)
            self.assertEqual(config['engine']['streams'],2)
            self.assertEqual(config['engine']['depth'],6)
            self.assertEqual(config['engine']['confidence'],0.70)
            self.assertEqual(config['engine']['kv_dtype'],'bf16')
            self.assertTrue(config['thinking_default'])
            self.assertEqual(config['vision_max_images'],64)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(serve.prefill_kernel_enabled())
        with patch.dict(os.environ, {'QWEN_PREFILL_KERNEL':'yes'}):
            with self.assertRaises(ValueError):serve.prefill_kernel_enabled()

    def test_launcher_does_not_inherit_an_accidental_host_switch(self):
        mod = load('supervise')
        args = [Path('/model'),Path('/vision/tower'),Path('/cache'),'owned',8001,Path('/cache/cid')]
        with patch.dict(os.environ, {'QWEN_PREFILL_KERNEL':'1'}):
            before=mod.docker_command(*args)
            after=mod.docker_command(*args,prefill_kernels=True)
        self.assertIn('QWEN_PREFILL_KERNEL=0',before)
        self.assertIn('QWEN_PREFILL_KERNEL=1',after)
        self.assertEqual([x for x in before if not x.startswith('QWEN_PREFILL_KERNEL=')], [x for x in after if not x.startswith('QWEN_PREFILL_KERNEL=')])
        self.assertIn('--memory-swap',after)
        self.assertNotIn('QWEN_PREFILL_READAHEAD=1',after)

    def test_dry_run_forwards_opt_in(self):
        mod=load('supervise')
        with contextlib.redirect_stdout(io.StringIO()) as out:
            mod.main(['/model','/tower','--prefill-kernels','--dry-run'])
        self.assertIn('QWEN_PREFILL_KERNEL=1',json.loads(out.getvalue()))

    def test_installed_service_preserves_the_switch(self):
        mod=load('install_service')
        self.assertNotIn('--prefill-kernels',mod.render('/model','/tower'))
        text=mod.render('/model','/tower',prefill_kernels=True)
        self.assertIn('"--prefill-kernels"',text)
        self.assertIn('MemorySwapMax=0',text)
        self.assertIn('Restart=no',text)

    def test_capacity_extra_is_accounted_without_context_or_policy_change(self):
        mod=load('supervise')
        self.assertEqual(mod.PROMPT_WORKSPACE_EXTRA,480285400)
        boundary={'mem_available':7*mod.GIB,'swap_used':mod.GIB,'disk_free':20*mod.GIB}
        self.assertIsNone(mod.breach(boundary,{'swap_used':0}))
        self.assertIsNotNone(mod.breach({**boundary,'mem_available':7*mod.GIB-1},{'swap_used':0}))
        self.assertIsNotNone(mod.breach({**boundary,'swap_used':mod.GIB+1},{'swap_used':0}))
        self.assertIsNotNone(mod.breach(boundary,{'swap_used':0},container_swap=1))

    def test_hashed_patch_is_installed_after_vision_patch(self):
        p=ROOT/'patches/prompt-expert-kernels.patch'
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        docker=(ROOT/'docker/Dockerfile').read_text()
        self.assertIn(sha,docker)
        self.assertIn(sha,(ROOT/'scripts/supervise.py').read_text())
        install='pip install --no-deps --no-build-isolation /opt/TensorFold'
        self.assertLess(docker.index('apply /opt/recipe/patches/vision-prefix-cache.patch'),docker.index('apply /opt/recipe/patches/prompt-expert-kernels.patch'))
        self.assertLess(docker.index('apply /opt/recipe/patches/prompt-expert-kernels.patch'),docker.index(install))
        source=p.read_text()
        self.assertIn('legacy212/*.cu',source)
        self.assertIn('legacy212/experts_grouped.cuh',source)
        self.assertNotIn('ngram_ahead.py',source)
        self.assertIn('prompt=b.prefill',source)
        self.assertIn('admission_extra',source)

    def test_public_patch_verifier_covers_legacy_and_prompt_sources(self):
        mod=load('verify_prefill_patch')
        self.assertEqual(len(mod.EXPECTED),18)
        self.assertEqual(len([x for x in mod.EXPECTED if '/legacy212/' in x]),6)
        self.assertIn('src/tensorfold/cuda/exl3/experts_prompt.cuh',mod.EXPECTED)
        self.assertIn('tests/cuda/test_exl3_prompt_experts.py',mod.EXPECTED)
        self.assertTrue(all(len(x)==64 for x in mod.EXPECTED.values()))


if __name__=='__main__':unittest.main()
