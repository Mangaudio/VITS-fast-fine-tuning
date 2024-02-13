<script setup>
</script>

<template>
  <main class="main-wrapper">
    <div class="grid" style="width: 100vw;">
      <div class="col-6 col-offset-3 text-center">
        <div class="flex flex-column">
          <div class="flex align-items-center justify-content-center h-4rem font-bold border-round m-2">
            输入需要转化为语音的文字（不超过150字符）：
          </div>
          <div class="flex align-items-center justify-content-center border-round m-2">
            <Textarea v-model="text" autoResize rows="5" cols="30" :maxlength="150"></Textarea>
          </div>
          <div class="flex align-items-center justify-content-center border-round m-2">
            <Dropdown v-model="model_name" :options="options" placeholder="请选择角色模型" class="border-round" />
          </div>
          <div class="flex align-items-center justify-content-center h-4rem border-round m-2">
            <div class="flex flex-wrap align-items-center mb-3 gap-2">
              <Button icon="pi pi-cog" @click="submit" class="h-3rem" :loading="loading" label="提交" />
              <a :href="audioSrc" download="audio.wav">
                <Button icon="pi pi-download" class="h-3rem" label="下载音频" link />
              </a>
            </div>
          </div>
          <div class="flex align-items-center justify-content-center h-4rem m-2">
            <audio controls></audio>
          </div>
        </div>
      </div>
    </div>
  </main>
</template>

<style scoped>
.main-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
}

.col-6 {
  grid-column: span 6;
}

.col-offset-3 {
  margin-left: auto;
  margin-right: auto;
}

.text-center {
  text-align: center;
}

.p-field {
  margin-bottom: 1rem;
}
</style>
<script setup>
import axios from 'axios';
import { ref, onMounted } from 'vue';

const text = ref('こんにちは');
const model_name = ref("");
const options = ref([]);
const loading = ref(false);
const audioSrc = ref("");

onMounted(async () => {
  try {
    loading.value = true;
    const response = await axios.get('/api/list');
    loading.value = false;
    options.value = response.data.models;
  } catch (error) {
    alert('获取角色模型列表失败，请刷新页面重试或联系管理员。')
    console.error(error);
  }
});

const submit = async () => {
  try {
    loading.value = true;
    const response = await axios.post('/api/tts', { text: text.value, model_name: model_name.value });
    const audio = document.querySelector('audio');
    if(response.data.error){
      alert(response.data.error);
      loading.value = false;
      return;
    }
    audio.src = response.data.file;
    audioSrc.value = response.data.file;
    audio.load();
    loading.value = false;
  } catch (error) {
    console.error(error);
  }
  
};

</script>
