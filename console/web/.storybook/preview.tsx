import type { Preview } from '@storybook/react-vite'
import { ConfigProvider } from 'antd'
import '../src/styles.css'

const preview: Preview = {
  decorators: [
    (Story) => <ConfigProvider theme={{ token: { colorPrimary: '#2563eb', borderRadius: 12 } }}><Story /></ConfigProvider>,
  ],
  parameters: {
    a11y: { test: 'error' },
    backgrounds: { default: 'console', values: [{ name: 'console', value: '#eef3f9' }] },
    layout: 'padded',
  },
}

export default preview
