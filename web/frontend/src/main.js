import 'primevue/resources/themes/aura-light-green/theme.css'
import 'primeflex/primeflex.css'
import 'primeicons/primeicons.css'

import { createApp } from 'vue'
import App from './App.vue'
import Button from 'primevue/button';
import Textarea from 'primevue/textarea';
import PrimeVue from "primevue/config";
import Dropdown from 'primevue/dropdown';
import { AVPlugin, AVWaveform } from "vue-audio-visual";


const app = createApp(App);
app.use(PrimeVue);
app.use(AVPlugin);
app.component('AVWaveform', AVWaveform);
app.component('Button', Button);
app.component('Textarea', Textarea);
app.component('Dropdown', Dropdown);
app.provide('value', 'Hello World')
app.mount('#app');
